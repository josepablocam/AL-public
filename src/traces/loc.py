import sqlite3

def get_lines(db_path, query):                                   
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  cursor.execute(query)
  scripts = map(lambda x: x[0], cursor.fetchall())
  # correct type
  scripts = filter(lambda x: isinstance(x, str) or isinstance(x, unicode), scripts)
  # split into lines
  script_lines = map(lambda x: x.split('\n'), scripts)
  # flatten, strip whitespace and remove empty lines
  script_lines = [line.strip() for lines in script_lines for line in lines if len(line.strip()) > 0]
  # not string (underestimate, since doesn't handle multiline strings)
  not_string = lambda x: x[0] != "'" and x[0] != '"'
  not_comment = lambda x: x[0] != "#"
  return filter(lambda x: not_string(x) and not_comment(x), script_lines)

meta_query = """ 
      select ScriptContent 
      from ScriptVersions, Scripts 
      where Scripts.CurrentScriptVersionId = ScriptVersions.Id 
      and ScriptVersions.ScriptLanguageId = (select Id from ScriptLanguages where Name = "Python") 
      group by ScriptContent
      """
      
meta_db = '../data/meta-kaggle/database.sqlite'
scraped_query = "select code from scripts"
scraped_db = '../data/crawled/kaggle.db'

print "Approx. %d LOC" % (len(get_lines(meta_db, meta_query)) + len(get_lines(scraped_db, scraped_query)))


