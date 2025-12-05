I have a task for you. You need to think thoroughly before coming up with an answer. In litellm , we have few tables whose data needs to be exported to BigQuery. Tables are LiteLLM_OrganizationTable LiteLLM_TeamTable LiteLLM_DailyTeamSpend All tables have createTime, so the cronjob should only pull incremental Write a python script that will do this task, this should be very dynamic in nature with parameters . The table structure might also change in future, so capability should be there, Creation, Updation, Deletion, Another point to remember this will work at 12:05 at night IST, with Kubernetes pod activating. Give me end to end scripts explaining everything Ask me if any doubt, don't assume anything on your own

Please note that parameterised in the sense also means that in future more tables can be added for scaling purposes

litellm-pg2bq-sync/
├── src/
│   └── litellm_sync/         ← package root
│       ├── __init__.py
│       └── sync.py            ← the main script logic (extracted from earlier)
├── config/                    ← optional: config files (e.g. template config.json)
│   └── config.json.template
├── Dockerfile
├── pyproject.toml
├── README.md
├── .gitignore

let's also create a pyproject.toml and all other relevant files for managing packages and the proper code repo
