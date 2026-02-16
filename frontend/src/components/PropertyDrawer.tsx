import { X, Euro, TrendingUp } from 'lucide-react';
import type { Property } from '../types';
import clsx from 'clsx';

interface PropertyDrawerProps {
    property: Property | null;
    onClose: () => void;
    isOpen: boolean;
}

export default function PropertyDrawer({ property, onClose, isOpen }: PropertyDrawerProps) {
    if (!isOpen) return null;

    // Loading state if property is null
    if (!property) {
        return (
            <div className={clsx(
                "fixed inset-y-0 right-0 w-96 bg-white shadow-2xl transform transition-transform duration-300 ease-in-out z-50 p-6 flex items-center justify-center",
                isOpen ? "translate-x-0" : "translate-x-full"
            )}>
                <div className="text-gray-500 flex flex-col items-center">
                    <TrendingUp className="animate-spin mb-2 text-blue-600" size={32} />
                    <span>Cargando detalles...</span>
                    <button onClick={onClose} className="mt-4 text-xs text-gray-400 hover:text-gray-600">Cerrar</button>
                </div>
            </div>
        );
    }

    const formatPrice = (price: number) => {
        return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 }).format(Number(price));
    };

    const formatPercent = (val: number) => {
        return new Intl.NumberFormat('es-ES', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(Number(val));
    };

    return (
        <div
            className={clsx(
                "fixed inset-y-0 right-0 w-96 bg-white shadow-2xl transform transition-transform duration-300 ease-in-out z-50 overflow-y-auto border-l border-gray-200",
                isOpen ? "translate-x-0" : "translate-x-full"
            )}
        >
            <div className="p-6">
                {/* Header */}
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <h2 className="text-xl font-bold text-gray-900">{property.title || 'Propiedad sin título'}</h2>
                        <p className="text-gray-500 text-sm mt-1">{property.neighborhood}, {property.district}</p>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                        <X size={24} />
                    </button>
                </div>

                {/* Key Metrics Grid */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="bg-blue-50 p-3 rounded-lg">
                        <div className="flex items-center space-x-2 text-blue-700 mb-1">
                            <Euro size={16} />
                            <span className="text-xs font-semibold uppercase">Precio</span>
                        </div>
                        <div className="text-lg font-bold text-blue-900">{formatPrice(property.price)}</div>
                        <div className="text-xs text-blue-600">{formatPrice(property.price_m2)}/m²</div>
                    </div>

                    <div className={clsx("p-3 rounded-lg", property.roi > 0 ? "bg-green-50" : "bg-red-50")}>
                        <div className={clsx("flex items-center space-x-2 mb-1", property.roi > 0 ? "text-green-700" : "text-red-700")}>
                            <TrendingUp size={16} />
                            <span className="text-xs font-semibold uppercase">ROI</span>
                        </div>
                        <div className={clsx("text-lg font-bold", property.roi > 0 ? "text-green-900" : "text-red-900")}>
                            {formatPercent(property.roi)}
                        </div>
                        <div className={clsx("text-xs", property.roi > 0 ? "text-green-600" : "text-red-600")}>
                            Rentabilidad estimada
                        </div>
                    </div>
                </div>

                {/* Investment Analysis */}
                <div className="mb-6">
                    <h3 className="text-sm font-bold text-gray-900 uppercase mb-3 border-b pb-1">Análisis de Inversión</h3>

                    <div className="space-y-3">
                        {/* Comprable Badge */}
                        {property.comprable === 1 && (
                            <div className="bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded w-fit mb-2">
                                Oportunidad de Compra (VI &gt; VO)
                            </div>
                        )}

                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <span className="text-gray-500 block text-xs">Prob. Venta (Inv)</span>
                                <span className="font-semibold text-gray-800">{formatPercent(property.cm)}</span>
                            </div>
                            <div>
                                <span className="text-gray-500 block text-xs">Prob. Venta (Prop)</span>
                                <span className="font-semibold text-gray-800">{formatPercent(property.ck)}</span>
                            </div>
                        </div>

                        {property.gap ? (
                            <div className="mt-2">
                                <span className="text-gray-500 block text-xs">Gap (CM/CK)</span>
                                <span className="font-semibold text-gray-800">{Number(property.gap).toFixed(2)}x</span>
                            </div>
                        ) : null}
                    </div>
                </div>

                {/* Details */}
                <div className="mb-6">
                    <h3 className="text-sm font-bold text-gray-900 uppercase mb-3 border-b pb-1">Detalles</h3>
                    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                        <dt className="text-gray-500">Superficie:</dt>
                        <dd className="font-medium text-gray-900">{property.size} m²</dd>

                        <dt className="text-gray-500">Habitaciones:</dt>
                        <dd className="font-medium text-gray-900">{property.bedrooms}</dd>

                        <dt className="text-gray-500">Baños:</dt>
                        <dd className="font-medium text-gray-900">{property.bathrooms}</dd>

                        <dt className="text-gray-500">Planta:</dt>
                        <dd className="font-medium text-gray-900">{property.floor || 'N/A'}</dd>
                    </dl>
                </div>

                {/* Amenities */}
                <div>
                    <h3 className="text-sm font-bold text-gray-900 uppercase mb-3 border-b pb-1">Amenities</h3>
                    <div className="flex flex-wrap gap-2">
                        {[
                            { key: 'lift', label: 'Ascensor' },
                            { key: 'terrace', label: 'Terraza' },
                            { key: 'garage', label: 'Parking' },
                            { key: 'storage', label: 'Trastero' },
                            { key: 'air_conditioning', label: 'Aire Acond.' },
                            { key: 'swimming_pool', label: 'Piscina' },
                            { key: 'garden', label: 'Jardín' },
                            { key: 'new_construction', label: 'Obra Nueva' }
                        ].map(amenity => (
                            // @ts-ignore
                            property[amenity.key] === 1 && (
                                <span key={amenity.key} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full border border-gray-200">
                                    {amenity.label}
                                </span>
                            )
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );
}
