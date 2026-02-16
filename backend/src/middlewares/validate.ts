import { Request, Response, NextFunction } from "express";
import { ZodSchema } from "zod";

export function validateQuery(schema: ZodSchema) {
    return (req: Request, res: Response, next: NextFunction) => {
        const out = schema.safeParse(req.query);
        if (!out.success) return res.status(400).json({ message: "Query inválida", issues: out.error.issues });
        req.query = out.data as any;
        next();
    };
}

export function validateBody(schema: ZodSchema) {
    return (req: Request, res: Response, next: NextFunction) => {
        const out = schema.safeParse(req.body);
        if (!out.success) return res.status(400).json({ message: "Body inválido", issues: out.error.issues });
        req.body = out.data;
        next();
    };
}
