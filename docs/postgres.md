# 🌈 PostgreSQL Database Maintenance

*Over time, as data is inserted, updated, and deleted, your PostgreSQL database—especially tables storing embeddings—can accumulate "dead tuples" and unused space. This can lead to increased disk usage and degraded performance. This document provides essential SQL commands to monitor database size and reclaim space.*

## Checking Database and Table Sizes

Regularly monitoring your database and table sizes helps in identifying potential issues early.

### Check size of all databases on the server
```sql
SELECT
    datname AS "Database",
    pg_size_pretty(pg_database_size(datname)) AS "Size"
FROM pg_database
ORDER BY pg_database_size(datname) DESC;
```
This query lists all databases on your PostgreSQL server and their respective sizes, ordered from largest to smallest.

### Check size of a specific table
```sql
SELECT
    relname AS "Table Name",
    pg_size_pretty(pg_table_size(C.oid)) AS "Data Size",
    pg_size_pretty(pg_indexes_size(C.oid)) AS "Index Size",
    pg_size_pretty(pg_total_relation_size(C.oid)) AS "Total Size"
FROM pg_class C
LEFT JOIN pg_namespace N ON (N.oid = C.relnamespace)
WHERE nspname NOT IN ('pg_catalog', 'information_schema')
  AND relkind <> 'i'
  AND nspname !~ '^pg_toast'
  AND relname = 'langchain_pg_embedding';
```
Replace `'langchain_pg_embedding'` with the name of the table you want to inspect. This query provides a detailed breakdown of the table's size, including data, indexes, and total size.

## Deleting Data

When you delete records, the space they occupied isn't immediately reclaimed. These operations create dead tuples that need to be cleaned up.

### Delete an embedded PDF (and its associated chunks)
```sql
-- This deletes all chunks belonging to a specific file
DELETE FROM langchain_pg_embedding
WHERE cmetadata->>'source' = '../TheEconomist_2504.pdf';
```
This command removes all embedding chunks associated with a specific source file, identified by its path in the `cmetadata` JSONB column.

### Drop a collection
To completely remove a collection and all its associated embeddings, follow these steps:

1.  **Find the UUID of your collection:**
    ```sql
    SELECT uuid FROM langchain_pg_collection WHERE name = 'your_old_collection_name';
    ```
    Replace `'your_old_collection_name'` with the actual name of the collection you wish to drop.

2.  **Delete the embeddings linked to that UUID:**
    ```sql
    DELETE FROM langchain_pg_embedding
    WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = 'your_old_collection_name');
    ```
    This step ensures all embedding vectors belonging to the specified collection are removed.

3.  **Delete the collection entry itself:**
    ```sql
    DELETE FROM langchain_pg_collection WHERE name = 'your_old_collection_name';
    ```
    Finally, this command removes the collection's entry from the `langchain_pg_collection` table.

## 🧹 Vacuuming and Disk Space Reclamation

`VACUUM` is crucial for managing disk space and improving query performance by cleaning up dead tuples.

### List dead tuples
```sql
SELECT
    schemaname AS "Schema",
    relname AS "Table",
    n_live_tup AS "Live Tuples",
    n_dead_tup AS "Dead Tuples",
    last_vacuum AS "Last Manual Vacuum",
    last_autovacuum AS "Last Auto-Vacuum"
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```
This query shows statistics for user tables, including the number of live and dead tuples, helping you identify tables that might benefit from vacuuming.

### Perform a VACUUM operation
```sql
-- VACUUM (ANALYZE): Reclaims space from dead tuples, but does not return it to the OS.
-- It updates statistics used by the query planner.
VACUUM (ANALYZE) langchain_pg_embedding;
VACUUM (ANALYZE) langchain_pg_collection;

-- VACUUM FULL (ANALYZE): Reclaims space more aggressively, returning it to the operating system.
-- This operation rewrites the entire table and requires an ACCESS EXCLUSIVE lock,
-- meaning the table will be unavailable for other operations during the vacuum.
-- Use with caution on production systems.
VACUUM FULL (ANALYZE) langchain_pg_embedding;
VACUUM FULL (ANALYZE) langchain_pg_collection;
```
Choose `VACUUM (ANALYZE)` for routine maintenance to clean up dead tuples and update statistics without blocking table access for long periods. Use `VACUUM FULL (ANALYZE)` when significant disk space needs to be reclaimed and you can tolerate downtime for the affected tables.