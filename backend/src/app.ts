import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { env } from "./config/env";
import { apiRouter } from "./routes/index";

export function createApp() {
    const app = express();

    app.use(express.json({ limit: "2mb" }));
    app.use(helmet());
    app.use(cors({
        origin: env.CORS_ORIGINS.length ? env.CORS_ORIGINS : false,
        credentials: true
    }));

    app.use("/api", rateLimit({
        windowMs: env.RATE_LIMIT_WINDOW_MS,
        max: env.RATE_LIMIT_MAX
    }));

    app.use(apiRouter);

    app.use((err: any, _req: any, res: any, _next: any) => {
        console.error(err);
        res.status(500).json({ message: "Error interno" });
    });

    return app;
}
