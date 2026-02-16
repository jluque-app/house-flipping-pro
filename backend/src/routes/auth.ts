import { Router } from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { pool } from "../db/pool";
import { env } from "../config/env";
import { validateBody } from "../middlewares/validate";
import { registerSchema, loginSchema } from "../schemas/common";

export const authRouter = Router();

authRouter.post("/register", validateBody(registerSchema), async (req, res) => {
    const { email, password } = req.body as any;
    const hash = await bcrypt.hash(password, 12);

    try {
        const r = await pool.query(
            `INSERT INTO users(email, password_hash, role) 
       VALUES ($1, $2, 'user') 
       RETURNING id, email, role`,
            [email.toLowerCase(), hash]
        );
        res.status(201).json(r.rows[0]);
    } catch (e: any) {
        if (String(e?.code) === "23505") return res.status(409).json({ message: "Email ya existe" });
        throw e;
    }
});

authRouter.post("/login", validateBody(loginSchema), async (req, res) => {
    const { email, password } = req.body as any;

    const r = await pool.query(
        `SELECT id, email, role, password_hash FROM users WHERE email = $1`,
        [email.toLowerCase()]
    );

    if (!r.rowCount) return res.status(401).json({ message: "Credenciales inválidas" });

    const user = r.rows[0];
    const ok = await bcrypt.compare(password, user.password_hash);
    if (!ok) return res.status(401).json({ message: "Credenciales inválidas" });

    const token = jwt.sign({ sub: String(user.id), role: user.role }, env.JWT_SECRET, { expiresIn: "2h" });
    res.json({ token });
});
