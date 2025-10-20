sudo systemctl daemon-reload
sudo systemctl enable lex-db-staging.service
sudo systemctl restart lex-db-staging.service
sudo systemctl status lex-db-staging.service --no-pager