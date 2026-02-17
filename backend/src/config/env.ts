import dotenv from "dotenv";
dotenv.config();

export const env = {
    NODE_ENV: process.env.NODE_ENV ?? "development",
    PORT: parseInt(process.env.PORT ?? "8080", 10),
    DATABASE_URL: process.env.DATABASE_URL ?? "",
    JWT_SECRET: process.env.JWT_SECRET ?? "",
    CORS_ORIGINS: [
        ...(process.env.CORS_ORIGINS ?? "").split(",").map(s => s.trim()).filter(Boolean),
        "https://investinhousing.eu",
        "https://www.investinhousing.eu",
        "https://house-flipping-pro.vercel.app"
    ],
    RATE_LIMIT_WINDOW_MS: parseInt(process.env.RATE_LIMIT_WINDOW_MS ?? "60000", 10),
    RATE_LIMIT_MAX: parseInt(process.env.RATE_LIMIT_MAX ?? "120", 10)
};

if (!env.DATABASE_URL) throw new Error("Falta DATABASE_URL");
if (!env.JWT_SECRET) throw new Error("Falta JWT_SECRET");
