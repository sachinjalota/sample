couple of things
1. I'm using external secrets for postgres connection string - projects/626624654237/secrets/LITELLM_SYNC_DEV_DATABASE_URL/versions/1 which has 'postgresql+psycopg2://postgres:GenAISept2025@localhost:5432/'
2. I don't want to install postgres in my docker rather it should connect with the postgres which is already on dev server 

So update yaml and dockerfile accordingly
