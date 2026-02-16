import { Router } from "express";
import { pool } from "../db/pool";
import { validateQuery } from "../middlewares/validate";
import { listQuerySchema } from "../schemas/common";
import { parseBBox } from "../utils/geo";

export const propertiesRouter = Router();

propertiesRouter.get("/", validateQuery(listQuerySchema), async (req, res) => {
  const q = req.query as any;
  const bbox = parseBBox(q.bbox);

  const limit = Math.min(q.limit ?? 2000, 10000);
  const offset = q.offset ?? 0;

  // filtros
  const where: string[] = [];
  const params: any[] = [];
  let i = 1;

  if (q.city) {
    where.push(`p.city = $${i++}`);
    params.push(q.city);
  }

  if (bbox) {
    where.push(`p.geom && ST_MakeEnvelope($${i++}, $${i++}, $${i++}, $${i++}, 4326)`);
    params.push(bbox[0], bbox[1], bbox[2], bbox[3]);
  }

  if (q.neighborhood) { where.push(`p.neighborhood = $${i++}`); params.push(q.neighborhood); }
  if (q.district) { where.push(`p.district = $${i++}`); params.push(q.district); }
  if (q.postal_code) { where.push(`p.postal_code = $${i++}`); params.push(q.postal_code); }

  if (q.price_min !== undefined) { where.push(`p.price >= $${i++}`); params.push(q.price_min); }
  if (q.price_max !== undefined) { where.push(`p.price <= $${i++}`); params.push(q.price_max); }

  if (q.roi_min !== undefined) { where.push(`p.roi >= $${i++}`); params.push(q.roi_min); }
  if (q.roi_max !== undefined) { where.push(`p.roi <= $${i++}`); params.push(q.roi_max); }

  if (q.comprable !== undefined) { where.push(`p.comprable = $${i++}`); params.push(q.comprable); }

  // amenities=lift,terrace,... (solo allowlist)
  const allowedAmen = new Set(["lift", "garage", "storage", "terrace", "air_conditioning", "swimming_pool", "garden", "sports", "new_construction"]);
  if (q.amenities) {
    const items = String(q.amenities).split(",").map((s: string) => s.trim()).filter(Boolean);
    for (const a of items) {
      if (!allowedAmen.has(a)) return res.status(400).json({ message: `Amenity inválida: ${a}` });
      where.push(`p.${a} = 1`);
    }
  }

  const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";

  // GeoJSON FeatureCollection (ligero para mapa)
  const sql = `
    SELECT jsonb_build_object(
      'type', 'FeatureCollection',
      'features', COALESCE(jsonb_agg(
        jsonb_build_object(
          'type', 'Feature',
          'geometry', ST_AsGeoJSON(p.geom)::jsonb,
          'properties', jsonb_build_object(
            'id', p.id,
            'price', p.price,
            'price_m2', p.price_m2,
            'roi', p.roi,
            'bedrooms', p.bedrooms,
            'bathrooms', p.bathrooms,
            'cm', p.cm,
            'ck', p.ck,
            'gap', (p.cm / NULLIF(p.ck,0)),
            'effective_price', (p.price_m2 * p.cm),
            'neighborhood', p.neighborhood,
            'district', p.district
          )
        )
      ), '[]'::jsonb)
    ) AS geojson
    FROM (
      SELECT *
      FROM properties p
      ${whereSql}
      ORDER BY p.id
      LIMIT $${i++} OFFSET $${i++}
    ) p;
  `;

  params.push(limit, offset);
  const r = await pool.query(sql, params);
  res.json(r.rows[0].geojson);
});

propertiesRouter.get("/:id", async (req, res) => {
  const id = Number(req.params.id);
  if (Number.isNaN(id)) return res.status(400).json({ message: "id inválido" });

  const sql = `
    SELECT 
      p.*,
      (p.cm / NULLIF(p.ck,0)) AS gap,
      (p.price_m2 * p.cm) AS effective_price
    FROM properties p
    WHERE p.id = $1
  `;
  const r = await pool.query(sql, [id]);

  if (!r.rowCount) return res.status(404).json({ message: "No encontrado" });
  res.json(r.rows[0]);
});
