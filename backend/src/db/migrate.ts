import fs from "fs";
import path from "path";
import { pool } from "./pool";

async function main() {
    await pool.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      id SERIAL PRIMARY KEY,
      filename TEXT UNIQUE NOT NULL,
      applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
  `);

    const dir = path.join(__dirname, "../../src/db/migrations");
    // Check if directory exists, if not try ../../db/migrations (depending on structure dist vs src)
    // But based on our structure it should be src/db/migrations or just ../../db/migrations relative to this file?
    // this file is in src/db/migrate.ts. So ../../src/db/migrations is weird.
    // It should be path.join(__dirname, "migrations");

    // Correction: The PDF says: const dir = path.join(__dirname, "../../db/migrations");
    // Let's stick to the PDF code but ensure the path is correct relative to execution.
    // If running via ts-node src/db/migrate.ts, __dirname is src/db.
    // So migrations are in src/db/migrations.

    const migrationDir = path.join(__dirname, "migrations");

    if (!fs.existsSync(migrationDir)) {
        console.error(`Migration directory not found: ${migrationDir}`);
        process.exit(1);
    }

    const files = fs.readdirSync(migrationDir).filter(f => f.endsWith(".sql")).sort();

    for (const f of files) {
        const already = await pool.query(
            "SELECT 1 FROM schema_migrations WHERE filename = $1",
            [f]
        );

        if (already.rowCount) continue;

        const sql = fs.readFileSync(path.join(migrationDir, f), "utf8");
        console.log("Applying migration:", f);

        await pool.query("BEGIN");
        try {
            await pool.query(sql);
            await pool.query("INSERT INTO schema_migrations(filename) VALUES ($1)", [f]);
            await pool.query("COMMIT");
        } catch (e) {
            await pool.query("ROLLBACK");
            throw e;
        }
    }

    console.log("Migrations OK");
    await pool.end();
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
