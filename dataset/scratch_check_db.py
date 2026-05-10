import os, psycopg2; from dotenv import load_dotenv; load_dotenv(); 
conn=psycopg2.connect(os.environ['DATABASE_URL']); cur=conn.cursor(); 
cur.execute("SELECT trail_type, count(*) FROM trail GROUP BY trail_type"); 
for row in cur.fetchall(): print(row)
conn.close()
