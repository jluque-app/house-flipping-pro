import { Request, Response, NextFunction } from "express";
import jwt from "jsonwebtoken";
import { env } from "../config/env";

export type JwtUser = { sub: string; role: "user" | "admin" };

declare global {
    namespace Express {
        interface Request { user?: JwtUser; }
    }
}

export function requireAuth(req: Request, res: Response, next: NextFunction) {
    const h = req.headers.authorization ?? "";
    const token = h.startsWith("Bearer ") ? h.slice(7) : null;

    if (!token) return res.status(401).json({ message: "No autenticado" });

    try {
        req.user = jwt.verify(token, env.JWT_SECRET) as JwtUser;
        return next();
    } catch {
        return res.status(401).json({ message: "Token inválido/expirado" });
    }
}

export function requireRole(role: "user" | "admin") {
    return (req: Request, res: Response, next: NextFunction) => {
        if (!req.user) return res.status(401).json({ message: "No autenticado" });
        if (req.user.role !== role) return res.status(403).json({ message: "Prohibido" });
        next();
    };
}
