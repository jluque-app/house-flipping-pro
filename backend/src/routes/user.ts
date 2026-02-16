import { Router } from "express";
import { pool } from "../db/pool";
import { requireAuth } from "../middlewares/auth";

export const userRouter = Router();

userRouter.get("/favorites", requireAuth, async (req, res) => {
    const userId = Number(req.user!.sub);
    const r = await pool.query(
        `SELECT f.property_id, p.title, p.neighborhood, p.price, p.roi 
     FROM favorites f
     JOIN properties p ON p.id = f.property_id
     WHERE f.user_id = $1
     ORDER BY f.created_at DESC`,
        [userId]
    );
    res.json(r.rows);
});

userRouter.post("/favorites/:propertyId", requireAuth, async (req, res) => {
    const userId = Number(req.user!.sub);
    const propertyId = Number(req.params.propertyId);
    if (Number.isNaN(propertyId)) return res.status(400).json({ message: "propertyId inválido" });

    await pool.query(
        `INSERT INTO favorites(user_id, property_id) VALUES ($1, $2)
     ON CONFLICT DO NOTHING`,
        [userId, propertyId]
    );
    res.status(204).send();
});

userRouter.delete("/favorites/:propertyId", requireAuth, async (req, res) => {
    const userId = Number(req.user!.sub);
    const propertyId = Number(req.params.propertyId);
    if (Number.isNaN(propertyId)) return res.status(400).json({ message: "propertyId inválido" });

    await pool.query(
        `DELETE FROM favorites WHERE user_id = $1 AND property_id = $2`,
        [userId, propertyId]
    );
    res.status(204).send();
});
