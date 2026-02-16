import { useState, useEffect } from 'react';
import { Map, BarChart3, TrendingUp, CheckCircle } from 'lucide-react';
import clsx from 'clsx';

interface OnboardingModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function OnboardingModal({ isOpen, onClose }: OnboardingModalProps) {
    const [step, setStep] = useState(0);

    // Reset step when opened
    useEffect(() => {
        if (isOpen) setStep(0);
    }, [isOpen]);

    const steps = [
        {
            icon: <Map className="w-12 h-12 text-blue-500" />,
            title: "Bienvenido a Inversión BCN",
            description: "Tu herramienta profesional para encontrar oportunidades de House Flipping en Barcelona."
        },
        {
            icon: <BarChart3 className="w-12 h-12 text-purple-500" />,
            title: "Analiza el ROI",
            description: "Cada propiedad incluye un cálculo estimado de rentabilidad basado en costes de reforma y precios de venta de la zona."
        },
        {
            icon: <TrendingUp className="w-12 h-12 text-green-500" />,
            title: "Entiende el Gap",
            description: "El 'Gap' te indica el margen de negociación. Buscamos propiedades donde el Valor de Inversión (VI) supere al Valor de Oferta (VO)."
        }
    ];

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 font-sans">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden animate-in zoom-in-95 duration-200">
                <div className="p-6 text-center">
                    <div className="flex justify-center mb-6">
                        <div className="bg-gray-50 p-4 rounded-full">
                            {steps[step].icon}
                        </div>
                    </div>

                    <h2 className="text-2xl font-bold text-gray-900 mb-2">{steps[step].title}</h2>
                    <p className="text-gray-600 mb-8 leading-relaxed">
                        {steps[step].description}
                    </p>

                    <div className="flex justify-center space-x-2 mb-8">
                        {steps.map((_, idx) => (
                            <div
                                key={idx}
                                className={clsx(
                                    "w-2 h-2 rounded-full transition-all duration-300",
                                    idx === step ? "bg-blue-600 w-6" : "bg-gray-200"
                                )}
                            />
                        ))}
                    </div>

                    <button
                        onClick={() => {
                            if (step < steps.length - 1) {
                                setStep(step + 1);
                            } else {
                                onClose();
                            }
                        }}
                        className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition-colors flex items-center justify-center"
                    >
                        {step < steps.length - 1 ? "Siguiente" : "Comenzar a Invertir"}
                        {step === steps.length - 1 && <CheckCircle className="ml-2 w-5 h-5" />}
                    </button>

                    <button
                        onClick={onClose}
                        className="mt-4 text-xs text-gray-400 hover:text-gray-600 uppercase font-semibold tracking-wide"
                    >
                        Saltar Tutorial
                    </button>
                </div>
            </div>
        </div>
    );
}
