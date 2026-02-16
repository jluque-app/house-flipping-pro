import { Router } from "express";
import { pool } from "../db/pool";
import { validateQuery } from "../middlewares/validate";
import { rankingQuerySchema } from "../schemas/common";
import { parseBBox } from "../utils/geo";

export const rankingRouter = Router();

rankingRouter.get("/", validateQuery(rankingQuerySchema), async (req, res) => {
    const q = req.query as any;
    const scope = q.scope ?? "neighborhood";
    const mode = q.mode ?? "roi";

    // defaults por modo
    const defaultDir: Record<string, "asc" | "desc"> = {
        roi: "desc",
        gap: "desc",
        effective_price: "asc", // "valor" por default: más bajo mejor
        liquidity: "desc"
    };
    const direction = q.direction ?? defaultDir[mode];
    const limit = Math.min(q.limit ?? 50, 500);

    // ORDER BY seguro (whitelist)
    const orderExpr: Record<string, string> = {
        roi: "p.roi",
        gap: "(p.cm / NULLIF(p.ck,0))",
        effective_price: "(p.price_m2 * p.cm)",
        liquidity: "p.cm"
    };
    const order = orderExpr[mode];
    if (!order) return res.status(400).json({ message: "mode inválido" });

    const where: string[] = [];
    const params: any[] = [];
    let i = 1;

    if (scope === "neighborhood") {
        if (!q.neighborhood) return res.status(400).json({ message: "Falta neighborhood" });
        where.push(`p.neighborhood = $${i++}`);
        params.push(q.neighborhood);
    } else if (scope === "viewport") {
        const bbox = parseBBox(q.bbox);
        if (!bbox) return res.status(400).json({ message: "bbox requerido e inválido" });
        where.push(`p.geom && ST_MakeEnvelope($${i++}, $${i++}, $${i++}, $${i++}, 4326)`);
        params.push(bbox[0], bbox[1], bbox[2], bbox[3]);
    } else {
        return res.status(400).json({ message: "scope inválido" });
    }

    const whereSql = `WHERE ${where.join(" AND ")}`;

    const sql = `
    SELECT 
      ROW_NUMBER() OVER (ORDER BY ${order} ${direction})::int AS rank,
      p.id, p.title, p.neighborhood, p.district,
      p.price, p.price_m2, p.size, p.bedrooms, p.bathrooms,
      p.roi, p.comprable, p.cm, p.ck,
      (p.cm / NULLIF(p.ck,0)) AS gap,
      (p.price_m2 * p.cm) AS effective_price
    FROM properties p
    ${whereSql}
    ORDER BY ${order} ${direction}
    LIMIT $${i++}
  `;
    params.push(limit);

    const r = await pool.query(sql, params);
    res.json({ scope, mode, direction, limit, items: r.rows });
});
