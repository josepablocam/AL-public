if [ ! $# -eq 1 ]
then
  echo "Usage: ./create_db.sh <num-scripts> <official-kaggle-db>"
  exit 1
fi

crawled_db='crawled_kaggle.db'
official_db='../../data/meta-kaggle/database.sqlite'
table_name='scripts'
table_definition="CREATE TABLE IF NOT EXISTS ${table_name}(link text, user text, votes integer, language text, code text)"

echo "==> Creating sqlite3 database: ${crawled_db}"
sqlite3 $db_name "${table_definition}"

echo "==> Collecting kernels"
python crawler.py $crawled_db $table_name --language python --sort 'most votes' --populate --num $1

echo "===> Adding records to official kaggle database"
python3 add_to_db.py $official_db $crawled_db
