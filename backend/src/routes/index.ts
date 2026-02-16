import { Router } from "express";
import { healthRouter } from "./health";
import { propertiesRouter } from "./properties";
import { catalogRouter } from "./catalog";
import { rankingRouter } from "./ranking";
import { authRouter } from "./auth";
import { userRouter } from "./user";
import { chatRouter } from "./chat";

export const apiRouter = Router();

apiRouter.use("/health", healthRouter);
apiRouter.use("/api/properties", propertiesRouter);
apiRouter.use("/api/catalog", catalogRouter);
apiRouter.use("/api/ranking", rankingRouter);
apiRouter.use("/api/auth", authRouter);
apiRouter.use("/api/user", userRouter);
apiRouter.use("/api/chat", chatRouter);
