import { Router } from "express";
import { pool } from "../db/pool";

export const catalogRouter = Router();

catalogRouter.get("/neighborhoods", async (_req, res) => {
    const r = await pool.query(`
    SELECT neighborhood, COUNT(*)::int AS count
    FROM properties
    WHERE neighborhood IS NOT NULL
    GROUP BY neighborhood
    ORDER BY count DESC, neighborhood ASC
  `);
    res.json(r.rows);
});

catalogRouter.get("/districts", async (_req, res) => {
    const r = await pool.query(`
    SELECT district, COUNT(*)::int AS count
    FROM properties
    WHERE district IS NOT NULL
    GROUP BY district
    ORDER BY count DESC, district ASC
  `);
    res.json(r.rows);
});
