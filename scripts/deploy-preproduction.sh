#!/bin/bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reloading systemctl daemon"
sudo systemctl daemon-reload
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Enabling lex-db-preproduction service"
sudo systemctl enable lex-db-preproduction.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting lex-db-preproduction service"
sudo systemctl restart lex-db-preproduction.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking lex-db-preproduction service status"
sudo systemctl status lex-db-preproduction.service 
exit 0
