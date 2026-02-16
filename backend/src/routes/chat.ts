import { Router } from "express";

export const chatRouter = Router();

chatRouter.post("/", async (req, res) => {
    const { message } = req.body;
    const userText = (message || "").toLowerCase();

    // Simulated "AI" Logic (Stub)
    let reply = "Soy un asistente virtual en fase de entrenamiento. Pronto podré responderte con datos en tiempo real.";

    if (userText.includes("hola") || userText.includes("buenos días")) {
        reply = "¡Hola! Soy tu asistente de inversiones. ¿En qué zona de Barcelona estás interesado hoy?";
    } else if (userText.includes("roi") || userText.includes("rentabilidad")) {
        reply = "El ROI (Retorno de Inversión) promedio en Barcelona actualmente ronda el 5-7% para reformas integrales. Zonas como Nou Barris pueden ofrecer un ROI más alto, cerca del 9%.";
    } else if (userText.includes("gap")) {
        reply = "El 'Gap' es la diferencia entre lo que un inversor dispuesto a reformar pagaría (VI) y lo que un propietario particular espera (VO). Un Gap mayor a 1 indica una buena oportunidad de negociación.";
    } else if (userText.includes("barrio") || userText.includes("zona")) {
        reply = "Para flipping, te recomiendo mirar Horta-Guinardó y Sant Andreu. Tienen precios de entrada más accesibles y alta demanda de vivienda reformada.";
    } else if (userText.includes("precio")) {
        reply = "Los precios varían mucho. En Ciutat Vella estamos viendo €3000-€4000/m², mientras que en Sarrià superan los €6000/m². ¿Tienes un presupuesto en mente?";
    }

    // Simulate network delay
    setTimeout(() => {
        res.json({ reply });
    }, 800);
});
