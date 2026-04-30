docker run --name pgvector-db \
  -e POSTGRES_USER=neko \
  -e POSTGRES_PASSWORD=${DB_PASSWORD} \
  -e POSTGRES_DB=vector_db \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16