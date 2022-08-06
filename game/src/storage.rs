use rusqlite::{Connection, OpenFlags, backup::Backup };
use uuid::Uuid;

use std::time::Duration;
use std::path::PathBuf;
use std::sync::Once;
use std::env;
use std::fs;

static mut SINGLETON: Option<Connection> = None;
static INIT: Once = Once::new();


pub struct Database {
  connection: Connection
}

#[derive(Debug)]
pub struct Turn {
    pub rowid: i32,
    pub previd: i32,
    pub key : String,
    pub answer: String,
    pub evaluation: i32,
    pub pieces: i32,
    pub over: bool,
}

impl Default for Database {
  fn default() -> Self { 

      let uri = "file:chess?mode=memory&cache=shared";
      let flags = OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_READ_WRITE;

      INIT.call_once(|| {
          // run initialization here
          let mut conn = Connection::open_with_flags( uri, flags).expect("Failed to create in-memory database");

          let backup_filename = get_db_dir().join("chess.sqlite").to_path_buf().display().to_string();
          if let Ok(src_conn) = Connection::open_with_flags( &backup_filename, OpenFlags::SQLITE_OPEN_READ_ONLY) {
            if let Ok(backup) = Backup::new( &src_conn, &mut conn) {
              if let Err(e) = backup.run_to_completion(1000, Duration::ZERO, None) {
                println!("WARNING: failed to restore backup database. {}", e);
              }
            } else {
              println!("WARNING: unable to restore backup database");
            }
          } else {
            println!("WARNING: unable to open backup database");
          }
          

          conn.execute(
            "CREATE TABLE IF NOT EXISTS main.turn (
                rowid    INTEGER PRIMARY KEY AUTOINCREMENT,
                previd INTEGER NOT NULL DEFAULT 0,
                key    TEXT NOT NULL,
                answer TEXT NOT NULL,
                evaluation  INTEGER NULL DEFAULT 0,
                pieces INTEGER NULL DEFAULT 0,
                over INTEGER NULL DEFAULT 0
            );",
            (), // empty list of parameters.
        ).expect("Failed to create table");

        conn.execute(
          "CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_key ON turn ('key');",
          (),
        ).expect("Failed to create index");

        conn.execute(
          "CREATE INDEX IF NOT EXISTS idx_turn_pieces_over ON turn ('pieces', 'over');",
          (),
        ).expect("Failed to create index");

        conn.execute(
          "CREATE INDEX IF NOT EXISTS idx_turn_previd ON turn ('previd', 'evaluation');",
          (),
        ).expect("Failed to create index");

        unsafe {
          SINGLETON = Some(conn);
        }
      });

    

      Database {
        connection : Connection::open_with_flags( uri, flags).expect("Failed to create in-memory database")
      }
   }
}


impl Database {

    pub fn store(self : &Self) {
      let dir = get_db_dir();
      fs::create_dir_all(&dir).expect(&format!("Unable to create db directory at {}", &dir.display()));

      let db_filename = dir.join(format!("{}.sqlite", Uuid::new_v4())).to_path_buf().display().to_string();
      {
        let mut dst_conn = Connection::open_with_flags( &db_filename, OpenFlags::SQLITE_OPEN_CREATE | OpenFlags::SQLITE_OPEN_READ_WRITE).expect("Failed to create database");
        let backup = Backup::new( &self.connection, &mut dst_conn).expect("Failed to create backup");
        backup.run_to_completion(1000, Duration::ZERO, None).expect("Failed to backup database");
      }

      let backup_filename = dir.join("chess.sqlite").to_path_buf().display().to_string();
      fs::remove_file(&backup_filename).unwrap_or_default();
      fs::rename(&db_filename, &backup_filename).expect("Failed to move file");
      //println!("{}", &backup_filename);
    }

    pub fn insert_only(self : &Self, turn : &Turn) {
      self.connection.execute(
        "INSERT OR IGNORE INTO main.turn (previd, key, answer, evaluation, pieces, over) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        (&turn.previd, &turn.key, &turn.answer, &turn.evaluation, &turn.pieces, &turn.over)
      ).expect("Failed to insert record");
    }

    pub fn set_over(self : &Self, rowid : i32) {
      self.connection.execute(
        "UPDATE main.turn SET over=?1 WHERE rowid=?2",
        (&rowid, true)
      ).expect("Failed to update record");
    }

    pub fn query_by_pieces(self : &Self, pieces : i32) -> Vec<Turn> {
      let mut stmt = self.connection.prepare("SELECT rowid, previd, key, answer, evaluation, pieces, over FROM main.turn WHERE pieces = ?1 AND over = 0").expect("Failed to query");
      
      stmt.query_map([pieces], |row| {
          Ok(Turn {
              rowid: row.get(0)?,
              previd: row.get(1)?,
              key: row.get(2)?,
              answer: row.get(3)?,
              evaluation: row.get(4)?,
              pieces: row.get(5)?,
              over : row.get(6)?,
          })
      })
      .expect("Failed to query database")
      .into_iter().map( |result| {
        result.unwrap()
      }).collect()
    }


    pub fn query_by_previd(self : &Self, previd : i32) -> Vec<Turn> {
      let mut stmt = self.connection.prepare("SELECT rowid, previd, key, answer, evaluation, pieces, over FROM main.turn WHERE previd = ?1").expect("Failed to query");
      
      stmt.query_map([previd], |row| {
          Ok(Turn {
              rowid: row.get(0)?,
              previd: row.get(1)?,
              key: row.get(2)?,
              answer: row.get(3)?,
              evaluation: row.get(4)?,
              pieces: row.get(5)?,
              over : row.get(6)?,
          })
      })
      .expect("Failed to query database")
      .into_iter().map( |result| {
        result.unwrap()
      }).collect()
    }
}



#[test]
fn test_db()
{
  let db = Database::default();
  db.store();
}


fn get_db_dir() -> PathBuf  {
  let filepath = env::current_exe().unwrap();
  if let Some(dir) = filepath.parent() {
      return dir.join("db");
  } else {
      panic!("Cannot find the file path.");
  }
}


