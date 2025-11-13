#!/bin/bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reloading systemctl daemon"
sudo systemctl daemon-reload
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Enabling lex-db-staging service"
sudo systemctl enable lex-db-staging.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting lex-db-staging service"
sudo systemctl restart lex-db-staging.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking lex-db-staging service status"
sudo systemctl status lex-db-staging.service 
exit 0