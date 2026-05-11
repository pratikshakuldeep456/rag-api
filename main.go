package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"github.com/pgvector/pgvector-go"
	_ "github.com/pgvector/pgvector-go/pgx"
)

const (
	embedDim              = 768
	defaultTopK           = 5
	maxRetrievalUserTurns = 4
	defaultHTTPTimeout    = 60 * time.Second
)

type createDocumentRequest struct {
	Text string `json:"text" binding:"required"`
}

type createDocumentResponse struct {
	ID int64 `json:"id"`
}

type queryRequest struct {
	Query string `json:"query" binding:"required"`
	TopK  int    `json:"top_k"`
}

type queryResponse struct {
	Answer  string   `json:"answer"`
	Context []string `json:"context"`
}

type chatMessage struct {
	Role    string `json:"role" binding:"required"`
	Content string `json:"content" binding:"required"`
}

type chatRequest struct {
	// Legacy: full thread in order (each turn user/assistant/…).
	Messages []chatMessage `json:"messages"`
	// Follow-up style: prior turns plus the new user question in message.
	History []chatMessage `json:"history"`
	Message string        `json:"message"`
	TopK    int           `json:"top_k"`
}

type chatResponse struct {
	Message string   `json:"message"`
	Role    string   `json:"role"`
	Context []string `json:"context"`
}

type ollamaEmbeddingsRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbeddingsResponse struct {
	Embedding []float32 `json:"embedding"`
}

type ollamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

type ollamaChatResponse struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
}

type appConfig struct {
	DBURL      string
	OllamaURL  string
	EmbedModel string
	ChatModel  string
	Port       string
}

func main() {
	_ = godotenv.Load()

	cfg := mustLoadConfig()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	poolCfg, err := pgxpool.ParseConfig(cfg.DBURL)
	if err != nil {
		log.Fatalf("db parse config: %v", err)
	}
	// Supabase pooled connections (PgBouncer transaction mode) break pgx prepared
	// statement caching and can return: prepared statement "...stmtcache..." already exists (42P05).
	poolCfg.ConnConfig.DefaultQueryExecMode = pgx.QueryExecModeSimpleProtocol

	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		log.Fatalf("db connect: %v", err)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		log.Fatalf("db ping: %v", err)
	}
	log.Printf("db connected")

	httpClient := &http.Client{Timeout: defaultHTTPTimeout}

	r := gin.Default()

	r.POST("/documents", func(c *gin.Context) {
		reqStart := time.Now()
		var req createDocumentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			log.Printf("POST /documents bind error: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
			return
		}
		text := strings.TrimSpace(req.Text)
		if text == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "text is required"})
			return
		}
		log.Printf("POST /documents received chars=%d", len(text))

		t0 := time.Now()
		emb, err := generateEmbedding(c.Request.Context(), httpClient, cfg, text)
		if err != nil {
			log.Printf("POST /documents embedding error: %v", err)
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		log.Printf("POST /documents embedding ok in %s", time.Since(t0))

		t1 := time.Now()
		id, err := insertDocument(c.Request.Context(), pool, text, emb)
		if err != nil {
			log.Printf("POST /documents insert error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to insert document", "details": err.Error()})
			return
		}
		log.Printf("POST /documents insert ok id=%d in %s total=%s", id, time.Since(t1), time.Since(reqStart))

		c.JSON(http.StatusOK, createDocumentResponse{ID: id})
	})

	r.POST("/query", func(c *gin.Context) {
		reqStart := time.Now()
		var req queryRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			log.Printf("POST /query bind error: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
			return
		}
		q := strings.TrimSpace(req.Query)
		if q == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "query is required"})
			return
		}
		log.Printf("POST /query received chars=%d top_k=%d", len(q), req.TopK)

		topK := req.TopK
		if topK <= 0 {
			topK = defaultTopK
		}
		if topK > 20 {
			topK = 20
		}

		t0 := time.Now()
		qEmb, err := generateEmbedding(c.Request.Context(), httpClient, cfg, normalizeRetrievalQuery(q))
		if err != nil {
			log.Printf("POST /query embedding error: %v", err)
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		log.Printf("POST /query embedding ok in %s", time.Since(t0))

		t1 := time.Now()
		ctxTexts, err := similaritySearch(c.Request.Context(), pool, qEmb, topK)
		if err != nil {
			log.Printf("POST /query similarity search error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to search documents", "details": err.Error()})
			return
		}
		log.Printf("POST /query similarity search ok results=%d in %s", len(ctxTexts), time.Since(t1))

		t2 := time.Now()
		prompt := buildPrompt(ctxTexts, q)
		log.Printf("POST /query prompt built chars=%d in %s", len(prompt), time.Since(t2))

		t3 := time.Now()
		msgs := []ollamaChatMessage{{Role: "user", Content: prompt}}
		ans, err := ollamaChatComplete(c.Request.Context(), httpClient, cfg, msgs)
		if err != nil {
			log.Printf("POST /query llama error: %v", err)
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		log.Printf("POST /query llama ok in %s total=%s", time.Since(t3), time.Since(reqStart))

		c.JSON(http.StatusOK, queryResponse{
			Answer:  strings.TrimSpace(ans),
			Context: ctxTexts,
		})
	})

	r.POST("/chat", func(c *gin.Context) {
		reqStart := time.Now()
		var req chatRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			log.Printf("POST /chat bind error: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
			return
		}
		thread, err := parseChatThread(req)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		lastUser, err := lastUserContent(thread)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		retrievalQuery := retrievalQueryFromMessages(thread, maxRetrievalUserTurns)
		retrievalQuery = normalizeRetrievalQuery(retrievalQuery)
		mode := "messages"
		if strings.TrimSpace(req.Message) != "" && len(req.Messages) == 0 {
			mode = "history+message"
		}
		log.Printf("POST /chat mode=%s turns=%d last_user_chars=%d retrieval_chars=%d top_k=%d", mode, len(thread), len(lastUser), len(retrievalQuery), req.TopK)

		topK := req.TopK
		if topK <= 0 {
			topK = defaultTopK
		}
		if topK > 20 {
			topK = 20
		}

		t0 := time.Now()
		qEmb, err := generateEmbedding(c.Request.Context(), httpClient, cfg, retrievalQuery)
		if err != nil {
			log.Printf("POST /chat embedding error: %v", err)
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		log.Printf("POST /chat embedding ok in %s", time.Since(t0))

		t1 := time.Now()
		ctxTexts, err := similaritySearch(c.Request.Context(), pool, qEmb, topK)
		if err != nil {
			log.Printf("POST /chat similarity search error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to search documents", "details": err.Error()})
			return
		}
		log.Printf("POST /chat similarity search ok results=%d in %s", len(ctxTexts), time.Since(t1))

		ollamaMsgs := []ollamaChatMessage{
			{Role: "system", Content: buildRAGSystemPrompt(ctxTexts)},
		}
		for _, m := range thread {
			role := normalizeChatRole(m.Role)
			if role == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "invalid message role: " + m.Role})
				return
			}
			ollamaMsgs = append(ollamaMsgs, ollamaChatMessage{Role: role, Content: m.Content})
		}

		t2 := time.Now()
		reply, err := ollamaChatComplete(c.Request.Context(), httpClient, cfg, ollamaMsgs)
		if err != nil {
			log.Printf("POST /chat ollama error: %v", err)
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		log.Printf("POST /chat ollama ok in %s total=%s", time.Since(t2), time.Since(reqStart))

		c.JSON(http.StatusOK, chatResponse{
			Message: strings.TrimSpace(reply),
			Role:    "assistant",
			Context: ctxTexts,
		})
	})

	log.Printf("listening on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatalf("server: %v", err)
	}
}

func mustLoadConfig() appConfig {
	cfg := appConfig{
		DBURL:      strings.TrimSpace(os.Getenv("SUPABASE_DB_URL")),
		OllamaURL:  strings.TrimSpace(os.Getenv("OLLAMA_URL")),
		EmbedModel: strings.TrimSpace(os.Getenv("EMBED_MODEL")),
		ChatModel:  strings.TrimSpace(os.Getenv("CHAT_MODEL")),
		Port:       strings.TrimSpace(os.Getenv("PORT")),
	}
	if cfg.DBURL == "" {
		log.Fatal("SUPABASE_DB_URL is required")
	}
	if cfg.OllamaURL == "" {
		cfg.OllamaURL = "http://localhost:11434"
	}
	if cfg.EmbedModel == "" {
		cfg.EmbedModel = "nomic-embed-text"
	}
	if cfg.ChatModel == "" {
		cfg.ChatModel = "llama3"
	}
	if cfg.Port == "" {
		cfg.Port = "8080"
	}
	// basic sanity check
	if _, err := strconv.Atoi(cfg.Port); err != nil {
		log.Fatalf("invalid PORT: %q", cfg.Port)
	}
	return cfg
}

func generateEmbedding(ctx context.Context, client *http.Client, cfg appConfig, text string) (pgvector.Vector, error) {
	url := strings.TrimRight(cfg.OllamaURL, "/") + "/api/embeddings"
	reqBody, _ := json.Marshal(ollamaEmbeddingsRequest{
		Model:  cfg.EmbedModel,
		Prompt: text,
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return pgvector.Vector{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return pgvector.Vector{}, fmt.Errorf("ollama embeddings request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return pgvector.Vector{}, fmt.Errorf("ollama embeddings returned %s from %s: %s", resp.Status, url, readSmallBody(resp.Body))
	}

	var out ollamaEmbeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return pgvector.Vector{}, fmt.Errorf("decode embeddings response: %w", err)
	}
	if len(out.Embedding) != embedDim {
		return pgvector.Vector{}, fmt.Errorf("unexpected embedding dimension: got %d want %d", len(out.Embedding), embedDim)
	}

	return pgvector.NewVector(out.Embedding), nil
}

func insertDocument(ctx context.Context, pool *pgxpool.Pool, text string, embedding pgvector.Vector) (int64, error) {
	var id int64
	err := pool.QueryRow(ctx,
		`INSERT INTO documents (content, embedding) VALUES ($1, $2) RETURNING id`,
		text, embedding,
	).Scan(&id)
	return id, err
}

func similaritySearch(ctx context.Context, pool *pgxpool.Pool, queryEmbedding pgvector.Vector, topK int) ([]string, error) {
	// Cosine distance (<=>): better match for sentence embeddings like nomic-embed-text than raw L2 (<->).
	rows, err := pool.Query(ctx,
		`SELECT content FROM documents ORDER BY embedding <=> $1 ASC LIMIT $2`,
		queryEmbedding, topK,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []string
	for rows.Next() {
		var content string
		if err := rows.Scan(&content); err != nil {
			return nil, err
		}
		out = append(out, content)
	}
	return out, rows.Err()
}

func buildRAGSystemPrompt(contexts []string) string {
	var b strings.Builder
	b.WriteString("You are a helpful assistant. The Context below is retrieved for this turn; snippets may be only partly relevant.\n")
	b.WriteString("If any snippet answers the user's question (even partially), synthesize a helpful answer and cite bracket numbers like [3] when you use them.\n")
	b.WriteString("Say you don't know only when no snippet contains usable information for the question.\n\nContext:\n")
	for i, ctx := range contexts {
		b.WriteString(fmt.Sprintf("[%d] %s\n", i+1, strings.TrimSpace(ctx)))
	}
	return b.String()
}

func buildPrompt(contexts []string, question string) string {
	var b strings.Builder
	b.WriteString(buildRAGSystemPrompt(contexts))
	b.WriteString("\nQuestion: ")
	b.WriteString(strings.TrimSpace(question))
	b.WriteString("\nAnswer:")
	return b.String()
}

func lastUserContent(msgs []chatMessage) (string, error) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if strings.EqualFold(strings.TrimSpace(msgs[i].Role), "user") {
			t := strings.TrimSpace(msgs[i].Content)
			if t == "" {
				return "", errors.New("last user message is empty")
			}
			return t, nil
		}
	}
	return "", errors.New("no user message found (end with a user turn for RAG)")
}

// parseChatThread supports:
//   - messages: full thread (legacy)
//   - history + message: prior turns + new user follow-up in message
func parseChatThread(req chatRequest) ([]chatMessage, error) {
	hasMsgs := len(req.Messages) > 0
	hasFollow := strings.TrimSpace(req.Message) != ""
	hasHist := len(req.History) > 0

	if hasMsgs && (hasFollow || hasHist) {
		return nil, errors.New("use either messages or history+message, not both")
	}
	if hasFollow {
		out := append([]chatMessage(nil), req.History...)
		out = append(out, chatMessage{Role: "user", Content: strings.TrimSpace(req.Message)})
		return out, nil
	}
	if hasMsgs {
		return req.Messages, nil
	}
	return nil, errors.New("provide messages or message (optional history for follow-up)")
}

// retrievalQueryFromMessages builds the text embedded for similarity search:
// concatenate the last maxUserTurns non-empty user messages (oldest first).
// Ensures entities from earlier turns (e.g. "Ollama") stay in scope for follow-ups.
func retrievalQueryFromMessages(msgs []chatMessage, maxUserTurns int) string {
	if maxUserTurns <= 0 {
		maxUserTurns = maxRetrievalUserTurns
	}
	var newestFirst []string
	for i := len(msgs) - 1; i >= 0 && len(newestFirst) < maxUserTurns; i-- {
		if !strings.EqualFold(strings.TrimSpace(msgs[i].Role), "user") {
			continue
		}
		t := strings.TrimSpace(msgs[i].Content)
		if t == "" {
			continue
		}
		newestFirst = append(newestFirst, t)
	}
	var b strings.Builder
	for i := len(newestFirst) - 1; i >= 0; i-- {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(newestFirst[i])
		if b.Len() >= 6000 {
			break
		}
	}
	return b.String()
}

func normalizeRetrievalQuery(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	var b strings.Builder
	for _, r := range s {
		if r == '\n' || r == '\r' {
			b.WriteRune('\n')
			continue
		}
		b.WriteRune(r)
	}
	return strings.TrimSpace(b.String())
}

func normalizeChatRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system":
		return "system"
	case "user":
		return "user"
	case "assistant":
		return "assistant"
	default:
		return ""
	}
}

func ollamaChatComplete(ctx context.Context, client *http.Client, cfg appConfig, messages []ollamaChatMessage) (string, error) {
	url := strings.TrimRight(cfg.OllamaURL, "/") + "/api/chat"
	reqBody, _ := json.Marshal(ollamaChatRequest{
		Model:    cfg.ChatModel,
		Messages: messages,
		Stream:   false,
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("ollama chat returned %s from %s: %s", resp.Status, url, readSmallBody(resp.Body))
	}

	var out ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}

	content := strings.TrimSpace(out.Message.Content)
	if content == "" {
		return "", errors.New("empty model response")
	}
	return content, nil
}

func readSmallBody(r io.Reader) string {
	const limit = 4096
	b, err := io.ReadAll(io.LimitReader(r, limit))
	if err != nil {
		return "<failed to read error body>"
	}
	s := strings.TrimSpace(string(b))
	if s == "" {
		return "<empty body>"
	}
	return s
}
