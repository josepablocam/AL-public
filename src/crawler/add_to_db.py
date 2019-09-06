from argparse import ArgumentParser
import sqlite3
import pandas as pd
import numpy as np
import sys


def insert_records(official, crawled):
    conn = sqlite3.connect(official)
    cursor = conn.cursor()
    # temporary table with python scripts only
    print "Setting up temporary table with existing Python code"
    cursor.execute(
        """
    create temp table pycode as
    select Scripts.Id, ScriptContent
              from ScriptVersions, Scripts
              where Scripts.CurrentScriptVersionId = ScriptVersions.Id
              and ScriptVersions.ScriptLanguageId in (select Id from ScriptLanguages where Name = "Python" or Name = "IPython Notebook")
              group by ScriptContent;
              """
    )
    cursor.execute("attach '%s' as crawled" % crawled)
    language_info = pd.read_sql_query(
        'select name as language, id as lang_id from scriptlanguages', conn
    )
    max_id = pd.read_sql_query('select max(id) from scripts',
                               conn).iloc[0][0] + 1
    not_dupes = pd.read_sql_query(
        """
    select language, code, votes from
    crawled.scripts where not code in (select ScriptContent from pycode) and
    language is not null
    """, conn
    )
    not_dupes['id'] = max_id + np.arange(not_dupes.shape[0])
    not_dupes = not_dupes.merge(language_info, on='language', how='left')

    recs = 0
    print "Inserting records"
    for _, row in not_dupes.iterrows():
        cursor.execute(
            "INSERT INTO ScriptVersions(Id, ScriptLanguageId, ScriptContent) VALUES(?, ?, ?)",
            tuple(row[['id', 'lang_id', 'code']])
        )
        cursor.execute(
            "INSERT INTO Scripts(CurrentScriptVersionId, TotalVotes) VALUES(?, ?)",
            tuple(row[['id', 'votes']])
        )
        recs += 1
    conn.commit()
    print("Inserted %d records" % recs)


if __name__ == "__main__":
    argparser = ArgumentParser(
        description="Insert non-duplicated records into main kaggle db"
    )
    argparser.add_argument(
        'official', type=str, help='Official meta-kaggle database path'
    )
    argparser.add_argument('crawled', type=str, help='Crawled database path')
    args = argparser.parse_args()
    insert_records(args.official, args.crawled)
