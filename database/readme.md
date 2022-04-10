```python
# create score.db
import sqlite3
connect = sqlite3.connect('score.db')
cursor = connect.cursor()
cursor.execute(
    'CREATE TABLE group_member_table'
    '(id INTEGER PRIMARY KEY AUTOINCREMENT,'
    'member_id INT,'
    'group_id INT,'
    'score INT,'
    'grade  INT)')
```