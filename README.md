(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % python -m litellm_sync.sync
<frozen runpy>:128: RuntimeWarning: 'litellm_sync.sync' found in sys.modules after import of package 'litellm_sync', but prior to execution of 'litellm_sync.sync'; this may result in unpredictable behaviour
2025-12-05 11:48:17,791 - __main__ - INFO - Connected to PostgreSQL
2025-12-05 11:48:18,280 - __main__ - INFO - Connected to BigQuery using Application Default Credentials
2025-12-05 11:48:18,490 - __main__ - INFO - Sync state table ensured
2025-12-05 11:48:18,490 - __main__ - INFO - Starting sync for 1 tables
2025-12-05 11:48:18,490 - __main__ - INFO - Starting sync for table: LiteLLM_OrganizationTable
2025-12-05 11:48:19,354 - __main__ - INFO - Last sync timestamp for LiteLLM_OrganizationTable: None
2025-12-05 11:48:19,544 - __main__ - INFO - Fetched 117 records from LiteLLM_OrganizationTable
2025-12-05 11:48:19,980 - __main__ - ERROR - Errors inserting rows to LiteLLM_OrganizationTable: [{'index': 0, 'errors': [{'reason': 'invalid', 'location': 'models', 'debugInfo': '', 'message': 'Array specified for non-repeated field: models.'}]}, {'index': 1, 'errors': [{'reason': 'invalid', 'location': 'metadata', 'debugInfo': '', 'message': 'This field: metadata is not a record.'}]}, {'index': 2, 'errors': [{'reason': 'invalid', 'location': 'model_spend', 'debugInfo': '', 'message': 'This field: model_spend is not a record.'}]}, {'index': 3, 'errors': [{'reason': 'invalid', 'location': 'models', 'debugInfo': '', 'message': 'Array specified for non-repeated field: models.'}]}, {'index': 4, 'errors': [{'reason': 'invalid', 'location': 'models', 'debugInfo': '', 'message': 'Array specified for non-repeated field: models.'}]}, {'index': 5, 'errors': [{'reason': 'invalid', 'location': 'models', 'debugInfo': '', 'message': 'Array specified for non-repeated field: models.'}]},
