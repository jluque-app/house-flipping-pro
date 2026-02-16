import { useState, useEffect } from 'react';
import { client } from '../api/client';
import { useFilters } from '../contexts/FilterContext';
import { useLanguage } from '../contexts/LanguageContext';
import type { Property } from '../types';
import { ArrowUpDown, ArrowDown } from 'lucide-react';
import Tooltip from '../components/Tooltip';

export default function RankingPage() {
    const { filters } = useFilters();
    const { t } = useLanguage();
    const [items, setItems] = useState<Property[]>([]);
    const [loading, setLoading] = useState(false);
    const [sortMode, setSortMode] = useState<'roi' | 'gap' | 'effective_price' | 'liquidity'>('roi');

    useEffect(() => {
        const fetchRanking = async () => {
            setLoading(true);
            try {
                const params = new URLSearchParams();
                params.append('mode', sortMode);
                params.append('limit', '50');

                // Scope handling: if filter has neighborhood, use neighborhood scope
                // otherwise default to neighborhood scope but require a neighborhood...
                // The API requires neighborhood if scope=neighborhood.
                // If no neighborhood selected, maybe we fallback to something or show empty? 
                // For now, let's default to a popular one or viewport if bbox exists.

                if (filters.neighborhood) {
                    params.append('scope', 'neighborhood');
                    params.append('neighborhood', filters.neighborhood);
                } else if (filters.bbox) {
                    params.append('scope', 'viewport');
                    params.append('bbox', filters.bbox);
                } else {
                    // Fallback or empty state
                    // Let's not fetch if no scope defined
                    setItems([]);
                    setLoading(false);
                    return;
                }

                const res = await client.get(`/ranking?${params.toString()}`);
                setItems(res.data.items);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchRanking();
    }, [filters, sortMode]);

    const Th = ({ label, mode, tooltip }: { label: string, mode: typeof sortMode, tooltip?: string }) => (
        <th
            className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 group relative"
            onClick={() => setSortMode(mode)}
        >
            <div className="flex items-center space-x-1">
                <span>{label}</span>
                {tooltip && <Tooltip text={tooltip} />}
                {sortMode === mode && <ArrowDown size={14} />}
                {sortMode !== mode && <ArrowUpDown size={14} className="text-gray-300" />}
            </div>
        </th>
    );

    if (!filters.neighborhood && !filters.bbox) {
        return (
            <div className="p-8 text-center text-gray-500">
                Selecciona un barrio o mueve el mapa para ver el ranking.
            </div>
        );
    }

    return (
        <div className="p-6 h-full overflow-y-auto">
            <h1 className="text-2xl font-bold mb-4 text-gray-800">Ranking de Oportunidades</h1>

            {loading ? (
                <div className="text-blue-600 p-4">Cargando...</div>
            ) : (
                <div className="bg-white shadow border-b border-gray-200 sm:rounded-lg overflow-visible">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50 sticky top-0 z-10">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">#</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{t('col.property')}</th>
                                <Th label={t('col.price')} mode="effective_price" />
                                <Th label={t('col.roi')} mode="roi" tooltip={t('drawer.roi')} />
                                <Th label={t('col.gap')} mode="gap" tooltip={t('drawer.gap')} />
                                <Th label={t('col.liquidity')} mode="liquidity" tooltip={t('drawer.liquidity')} />
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {items.map((item, idx) => (
                                <tr key={item.id} className="hover:bg-gray-50">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{idx + 1}</td>
                                    <td className="px-6 py-4">
                                        <div className="text-sm font-medium text-gray-900 truncate max-w-xs">{item.title}</div>
                                        <div className="text-sm text-gray-500">{item.neighborhood}</div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                        {new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR' }).format(Number(item.price))}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${item.roi > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                                            {(Number(item.roi) * 100).toFixed(1)}%
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {item.gap ? Number(item.gap).toFixed(2) : '-'}x
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {(Number(item.cm) * 100).toFixed(1)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
