import { Info } from "lucide-react";
import React from "react";

interface TooltipProps {
    text: string;
    children?: React.ReactNode;
}

export default function Tooltip({ text, children }: TooltipProps) {
    return (
        <div className="group relative flex items-center">
            {children || <Info size={14} className="text-gray-400 hover:text-gray-600 cursor-help ml-1" />}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 text-center shadow-lg pointer-events-none after:content-[''] after:absolute after:top-full after:left-1/2 after:-translate-x-1/2 after:border-4 after:border-transparent after:border-t-gray-800">
                {text}
            </div>
        </div>
    );
}
