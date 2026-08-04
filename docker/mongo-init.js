const database = db.getSiblingDB(process.env.MONGO_INITDB_DATABASE || "audio_pipeline");

database.createUser({
  user: process.env.MONGO_APP_USERNAME || "audio_pipeline",
  pwd: process.env.MONGO_APP_PASSWORD,
  roles: [{ role: "readWrite", db: database.getName() }],
});

database.createCollection("audio_jobs");
database.audio_jobs.createIndex({ status: 1, updated_at: 1 });
database.audio_jobs.createIndex({ created_at: 1 });
