import { z } from "zod";

export const listQuerySchema = z.object({
    bbox: z.string().optional(),
    neighborhood: z.string().optional(),
    district: z.string().optional(),
    postal_code: z.string().optional(),
    price_min: z.coerce.number().optional(),
    price_max: z.coerce.number().optional(),
    roi_min: z.coerce.number().optional(),
    roi_max: z.coerce.number().optional(),
    comprable: z.coerce.number().int().optional(),
    amenities: z.string().optional(),
    limit: z.coerce.number().int().optional(),
    offset: z.coerce.number().int().optional()
});

export const rankingQuerySchema = z.object({
    scope: z.enum(["neighborhood", "viewport"]).optional(),
    neighborhood: z.string().optional(),
    bbox: z.string().optional(),
    mode: z.enum(["roi", "gap", "effective_price", "liquidity"]).optional(),
    direction: z.enum(["asc", "desc"]).optional(),
    limit: z.coerce.number().int().optional()
});

export const registerSchema = z.object({
    email: z.string().email(),
    password: z.string().min(8)
});

export const loginSchema = registerSchema;
