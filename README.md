"""
LiteLLM PostgreSQL to BigQuery Incremental Sync
Syncs tables incrementally based on created_at/updated_at timestamps
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

import psycopg2
from psycopg2.extras import RealDictCursor
from google.cloud import bigquery
from google.oauth2 import service_account


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SyncStateManager:
    """Manages sync state to track last sync timestamps"""
    
    def __init__(self, pg_conn):
        self.pg_conn = pg_conn
        self._ensure_state_table()
    
    def _ensure_state_table(self):
        """Create sync state table if not exists"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS litellm_sync_state (
            table_name VARCHAR(255) PRIMARY KEY,
            last_sync_timestamp TIMESTAMP,
            last_sync_status VARCHAR(50),
            last_sync_record_count INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.pg_conn.cursor() as cur:
            cur.execute(create_table_sql)
        self.pg_conn.commit()
        logger.info("Sync state table ensured")
    
    def get_last_sync_timestamp(self, table_name: str) -> Optional[datetime]:
        """Get last successful sync timestamp for a table"""
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT last_sync_timestamp FROM litellm_sync_state WHERE table_name = %s AND last_sync_status = 'SUCCESS'",
                (table_name,)
            )
            result = cur.fetchone()
            return result['last_sync_timestamp'] if result else None
    
    def update_sync_state(self, table_name: str, timestamp: datetime, status: str, record_count: int):
        """Update sync state for a table"""
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO litellm_sync_state (table_name, last_sync_timestamp, last_sync_status, last_sync_record_count, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (table_name) 
                DO UPDATE SET 
                    last_sync_timestamp = EXCLUDED.last_sync_timestamp,
                    last_sync_status = EXCLUDED.last_sync_status,
                    last_sync_record_count = EXCLUDED.last_sync_record_count,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (table_name, timestamp, status, record_count)
            )
        self.pg_conn.commit()


class LiteLLMBigQuerySync:
    """Main sync orchestrator"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = self._load_config(config_path)
        self.pg_conn = self._connect_postgres()
        self.bq_client = self._connect_bigquery()
        self.state_manager = SyncStateManager(self.pg_conn)
        self.sync_results = {}
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _connect_postgres(self) -> psycopg2.extensions.connection:
        """Connect to PostgreSQL"""
        try:
            conn = psycopg2.connect(self.config['postgres_connection_string'])
            logger.info("Connected to PostgreSQL")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _connect_bigquery(self) -> bigquery.Client:
        """Connect to BigQuery"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.config['bigquery_service_account_path']
            )
            client = bigquery.Client(
                credentials=credentials,
                project=self.config['bigquery_project']
            )
            logger.info("Connected to BigQuery")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            raise
    
    def _get_pg_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Get PostgreSQL table schema"""
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name.lower(),)
            )
            return cur.fetchall()
    
    def _pg_to_bq_type(self, pg_type: str) -> str:
        """Map PostgreSQL data types to BigQuery data types"""
        type_mapping = {
            'integer': 'INTEGER',
            'bigint': 'INTEGER',
            'smallint': 'INTEGER',
            'numeric': 'NUMERIC',
            'real': 'FLOAT64',
            'double precision': 'FLOAT64',
            'character varying': 'STRING',
            'varchar': 'STRING',
            'text': 'STRING',
            'character': 'STRING',
            'char': 'STRING',
            'boolean': 'BOOLEAN',
            'timestamp without time zone': 'TIMESTAMP',
            'timestamp with time zone': 'TIMESTAMP',
            'date': 'DATE',
            'time': 'TIME',
            'json': 'JSON',
            'jsonb': 'JSON',
            'uuid': 'STRING',
            'bytea': 'BYTES'
        }
        return type_mapping.get(pg_type.lower(), 'STRING')
    
    def _ensure_bq_table(self, table_name: str) -> bigquery.Table:
        """Ensure BigQuery table exists and has correct schema"""
        dataset_ref = self.bq_client.dataset(self.config['bigquery_dataset'])
        table_ref = dataset_ref.table(table_name)
        full_table_id = f"{self.config['bigquery_project']}.{self.config['bigquery_dataset']}.{table_name}"
        
        # Get PostgreSQL schema
        pg_schema = self._get_pg_table_schema(table_name)
        
        # Create BigQuery schema
        bq_schema = [
            bigquery.SchemaField(
                col['column_name'],
                self._pg_to_bq_type(col['data_type']),
                mode='NULLABLE'
            )
            for col in pg_schema
        ]
        
        try:
            # Try to get existing table
            existing_table = self.bq_client.get_table(table_ref)
            
            # Compare schemas and update if needed
            existing_fields = {field.name: field for field in existing_table.schema}
            new_fields = {field.name: field for field in bq_schema}
            
            # Check for new columns
            fields_to_add = [
                field for name, field in new_fields.items() 
                if name not in existing_fields
            ]
            
            if fields_to_add:
                logger.info(f"Adding {len(fields_to_add)} new columns to {table_name}")
                updated_schema = list(existing_table.schema) + fields_to_add
                existing_table.schema = updated_schema
                existing_table = self.bq_client.update_table(existing_table, ['schema'])
                logger.info(f"Schema updated for {table_name}")
            
            return existing_table
            
        except Exception as e:
            # Table doesn't exist, create it
            logger.info(f"Creating BigQuery table: {table_name}")
            table = bigquery.Table(table_ref, schema=bq_schema)
            
            # Add table options
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="updated_at"  # Partition by updated_at
            )
            
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {full_table_id}")
            return table
    
    def _fetch_incremental_data(self, table_name: str, last_sync_timestamp: Optional[datetime]) -> List[Dict]:
        """Fetch incremental data from PostgreSQL"""
        timestamp_columns = self.config.get('timestamp_columns', ['created_at', 'updated_at'])
        
        # Build WHERE clause for incremental sync
        if last_sync_timestamp:
            where_conditions = " OR ".join([
                f"{col} > %s" for col in timestamp_columns
            ])
            where_clause = f"WHERE {where_conditions}"
            params = tuple([last_sync_timestamp] * len(timestamp_columns))
        else:
            where_clause = ""
            params = ()
        
        query = f"SELECT * FROM {table_name} {where_clause}"
        
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            records = cur.fetchall()
        
        # Convert to list of dicts and handle datetime serialization
        result = []
        for record in records:
            row_dict = dict(record)
            # Convert datetime objects to ISO format strings
            for key, value in row_dict.items():
                if isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            result.append(row_dict)
        
        return result
    
    def _load_to_bigquery(self, table_name: str, records: List[Dict]) -> int:
        """Load records to BigQuery using streaming insert"""
        if not records:
            return 0
        
        table_ref = self.bq_client.dataset(self.config['bigquery_dataset']).table(table_name)
        
        # Use WRITE_TRUNCATE for full refresh, WRITE_APPEND for incremental
        # For upsert behavior, we need to use MERGE which requires a staging approach
        
        # For now, using streaming inserts with deduplication handled by updated_at
        errors = self.bq_client.insert_rows_json(table_ref, records)
        
        if errors:
            logger.error(f"Errors inserting rows to {table_name}: {errors}")
            # For better upsert behavior, we should use a MERGE statement
            # This requires writing to a temp table first, then merging
            return self._upsert_via_merge(table_name, records)
        
        return len(records)
    
    def _upsert_via_merge(self, table_name: str, records: List[Dict]) -> int:
        """Perform upsert using MERGE statement via temp table"""
        if not records:
            return 0
        
        # Create temp table name
        temp_table_name = f"{table_name}_temp_{int(time.time())}"
        dataset_ref = self.bq_client.dataset(self.config['bigquery_dataset'])
        temp_table_ref = dataset_ref.table(temp_table_name)
        
        try:
            # Get schema from main table
            main_table = self.bq_client.get_table(dataset_ref.table(table_name))
            
            # Create temp table
            temp_table = bigquery.Table(temp_table_ref, schema=main_table.schema)
            temp_table = self.bq_client.create_table(temp_table)
            
            # Load data to temp table
            job_config = bigquery.LoadJobConfig(
                schema=main_table.schema,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            
            job = self.bq_client.load_table_from_json(
                records, temp_table_ref, job_config=job_config
            )
            job.result()  # Wait for job to complete
            
            # Perform MERGE
            # Assuming primary key is the first column or a common 'id' column
            schema_fields = [field.name for field in main_table.schema]
            primary_key = self.config.get('primary_keys', {}).get(table_name, 'id')
            
            update_clause = ", ".join([
                f"target.{col} = source.{col}" 
                for col in schema_fields if col != primary_key
            ])
            
            insert_cols = ", ".join(schema_fields)
            insert_vals = ", ".join([f"source.{col}" for col in schema_fields])
            
            merge_query = f"""
            MERGE `{self.config['bigquery_project']}.{self.config['bigquery_dataset']}.{table_name}` AS target
            USING `{self.config['bigquery_project']}.{self.config['bigquery_dataset']}.{temp_table_name}` AS source
            ON target.{primary_key} = source.{primary_key}
            WHEN MATCHED THEN
                UPDATE SET {update_clause}
            WHEN NOT MATCHED THEN
                INSERT ({insert_cols})
                VALUES ({insert_vals})
            """
            
            query_job = self.bq_client.query(merge_query)
            query_job.result()
            
            logger.info(f"MERGE completed for {table_name}")
            
            return len(records)
            
        finally:
            # Clean up temp table
            self.bq_client.delete_table(temp_table_ref, not_found_ok=True)
    
    def sync_table(self, table_name: str, retry_count: int = 0) -> Dict[str, Any]:
        """Sync a single table from PostgreSQL to BigQuery"""
        max_retries = self.config.get('max_retries', 2)
        
        try:
            logger.info(f"Starting sync for table: {table_name}")
            
            # Ensure BigQuery table exists
            self._ensure_bq_table(table_name)
            
            # Get last sync timestamp
            last_sync = self.state_manager.get_last_sync_timestamp(table_name)
            logger.info(f"Last sync timestamp for {table_name}: {last_sync}")
            
            # Fetch incremental data
            records = self._fetch_incremental_data(table_name, last_sync)
            logger.info(f"Fetched {len(records)} records from {table_name}")
            
            if records:
                # Load to BigQuery
                loaded_count = self._load_to_bigquery(table_name, records)
                logger.info(f"Loaded {loaded_count} records to BigQuery for {table_name}")
                
                # Update sync state
                current_timestamp = datetime.now(timezone.utc)
                self.state_manager.update_sync_state(
                    table_name, current_timestamp, 'SUCCESS', loaded_count
                )
                
                return {
                    'status': 'SUCCESS',
                    'records_synced': loaded_count,
                    'last_sync': current_timestamp.isoformat()
                }
            else:
                logger.info(f"No new records to sync for {table_name}")
                return {
                    'status': 'SUCCESS',
                    'records_synced': 0,
                    'message': 'No new records'
                }
        
        except Exception as e:
            logger.error(f"Error syncing table {table_name}: {e}", exc_info=True)
            
            if retry_count < max_retries:
                logger.info(f"Retrying {table_name} (attempt {retry_count + 1}/{max_retries})")
                time.sleep(5)  # Wait 5 seconds before retry
                return self.sync_table(table_name, retry_count + 1)
            else:
                # Update state as failed
                self.state_manager.update_sync_state(
                    table_name, datetime.now(timezone.utc), 'FAILED', 0
                )
                return {
                    'status': 'FAILED',
                    'error': str(e),
                    'retry_count': retry_count
                }
    
    def sync_all_tables(self) -> Dict[str, Any]:
        """Sync all configured tables"""
        tables = self.config.get('tables', [])
        logger.info(f"Starting sync for {len(tables)} tables")
        
        results = {}
        for table_name in tables:
            results[table_name] = self.sync_table(table_name)
        
        # Summary
        success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        failed_count = len(results) - success_count
        total_records = sum(r.get('records_synced', 0) for r in results.values())
        
        summary = {
            'total_tables': len(tables),
            'successful': success_count,
            'failed': failed_count,
            'total_records_synced': total_records,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'table_results': results
        }
        
        logger.info(f"Sync completed: {success_count} successful, {failed_count} failed, {total_records} total records")
        
        return summary
    
    def close(self):
        """Close connections"""
        if self.pg_conn:
            self.pg_conn.close()
        if self.bq_client:
            self.bq_client.close()
        logger.info("Connections closed")


def main():
    """Main entry point"""
    try:
        # Initialize sync
        sync = LiteLLMBigQuerySync(config_path="config/config.json")
        
        # Run sync
        results = sync.sync_all_tables()
        
        # Log results
        logger.info(f"Sync results: {json.dumps(results, indent=2)}")
        
        # Close connections
        sync.close()
        
        # Exit with appropriate code
        if results['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
