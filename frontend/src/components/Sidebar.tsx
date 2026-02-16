import { useEffect, useState } from 'react';
import { Filter, Home, List, LogOut, HelpCircle, Globe } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useFilters } from '../contexts/FilterContext';
import { useLanguage } from '../contexts/LanguageContext';
import { client } from '../api/client';
import type { CatalogItem } from '../types';

interface SidebarProps {
    view: 'map' | 'ranking';
    onViewChange: (view: 'map' | 'ranking') => void;
    onOpenHelp: () => void;
}

export default function Sidebar({ view, onViewChange, onOpenHelp }: SidebarProps) {
    const { logout } = useAuth();
    const { filters, updateFilter, resetFilters } = useFilters();
    const { t, language, setLanguage } = useLanguage();
    const [city, setCity] = useState<'Barcelona' | 'Valencia' | 'Madrid'>('Barcelona');
    const [neighborhoods, setNeighborhoods] = useState<CatalogItem[]>([]);

    useEffect(() => {
        updateFilter('city', city);
    }, [city]);

    useEffect(() => {
        client.get('/catalog/neighborhoods')
            .then(res => setNeighborhoods(res.data))
            .catch(err => console.error("Error fetching neighborhoods", err));
    }, []);

    const handleInputChange = (key: string, value: any) => {
        updateFilter(key as any, value === '' ? undefined : value);
    };

    return (
        <div className="flex flex-col h-full w-80 bg-white shadow-xl z-20 flex-shrink-0">
            {/* Header */}
            <div className="p-4 border-b border-gray-200 bg-blue-600 text-white flex justify-between items-center shrink-0">
                <div>
                    <Link to="/" className="text-xl font-bold hover:underline">House Flipping Pro</Link>
                    <p className="text-xs opacity-80">{t('app.subtitle')}</p>
                </div>
                <button
                    onClick={() => setLanguage(language === 'es' ? 'en' : 'es')}
                    className="flex items-center space-x-1 text-xs bg-blue-700 hover:bg-blue-800 px-2 py-1 rounded transition-colors"
                >
                    <Globe size={12} />
                    <span className="uppercase">{language}</span>
                </button>
            </div>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 min-h-0">
                {/* Navigation */}
                <div className="space-y-2">
                    <button
                        onClick={() => onViewChange('map')}
                        className={`w-full flex items-center space-x-2 px-3 py-2 rounded-md transition-colors ${view === 'map' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        <Home size={20} />
                        <span>{t('nav.map')}</span>
                    </button>
                    <button
                        onClick={() => onViewChange('ranking')}
                        className={`w-full flex items-center space-x-2 px-3 py-2 rounded-md transition-colors ${view === 'ranking' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        <List size={20} />
                        <span>{t('nav.ranking')}</span>
                    </button>
                </div>

                {/* Filters */}
                <div className="border-t border-gray-200 pt-4">
                    <div className="flex items-center justify-between text-gray-900 mb-4">
                        <div className="flex items-center space-x-2">
                            <Filter size={16} />
                            <span className="text-sm font-bold uppercase tracking-wide">{t('nav.filters')}</span>
                        </div>
                        <button
                            onClick={resetFilters}
                            className="text-xs text-blue-600 hover:text-blue-800 underline"
                        >
                            {t('nav.clear')}
                        </button>
                    </div>

                    {/* City Selector */}
                    <div className="mb-4">
                        <label className="block text-xs font-semibold text-gray-500 mb-1">Ciudad</label>
                        <select
                            className="w-full text-sm border border-gray-300 rounded px-2 py-1.5 appearance-none focus:outline-none focus:border-blue-500 bg-white"
                            value={city}
                            onChange={(e) => setCity(e.target.value as any)}
                        >
                            <option value="Barcelona">Barcelona</option>
                            <option value="Valencia">Valencia</option>
                            <option value="Madrid">Madrid (Próximamente)</option>
                        </select>
                    </div>

                    {city === 'Madrid' ? (
                        <div className="p-4 bg-yellow-50 text-yellow-800 text-sm rounded-md border border-yellow-200">
                            <strong>Madrid</strong> -- forthcoming soon.
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {/* Neighborhood */}
                            <div>
                                <label className="block text-xs font-semibold text-gray-500 mb-1">{t('filter.neighborhood')}</label>
                                <div className="relative">
                                    <select
                                        className="w-full text-sm border border-gray-300 rounded px-2 py-1.5 appearance-none focus:outline-none focus:border-blue-500 bg-white"
                                        value={filters.neighborhood || ''}
                                        onChange={(e) => handleInputChange('neighborhood', e.target.value)}
                                    >
                                        <option value="">{t('filter.all_neighborhoods')}</option>
                                        {neighborhoods.map((n) => (
                                            <option key={n.neighborhood} value={n.neighborhood}>
                                                {n.neighborhood} ({n.count})
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            {/* Price Max */}
                            <div>
                                <label className="block text-xs font-semibold text-gray-500 mb-1">{t('filter.price')} (Max)</label>
                                <input
                                    type="number"
                                    className="w-full text-sm border border-gray-300 rounded px-2 py-1.5 focus:outline-none focus:border-blue-500"
                                    placeholder="500000"
                                    value={filters.price_max || ''}
                                    onChange={(e) => handleInputChange('price_max', Number(e.target.value))}
                                />
                            </div>

                            {/* ROI Min */}
                            <div>
                                <label className="block text-xs font-semibold text-gray-500 mb-1">{t('filter.roi_min')}</label>
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                    value={filters.roi_min ? filters.roi_min * 100 : 0}
                                    onChange={(e) => handleInputChange('roi_min', Number(e.target.value) / 100)}
                                />
                                <div className="text-right text-xs text-blue-600 font-bold mt-1">
                                    {filters.roi_min ? Math.round(filters.roi_min * 100) : 0}%
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-gray-200 bg-gray-50 space-y-2 shrink-0">
                <button
                    onClick={onOpenHelp}
                    className="flex items-center space-x-2 text-gray-600 hover:text-blue-600 w-full px-3 py-2 rounded-md hover:bg-gray-100 transition-colors"
                >
                    <HelpCircle size={20} />
                    <span className="text-sm font-medium">{t('nav.help')}</span>
                </button>
                <button
                    onClick={logout}
                    className="flex items-center space-x-2 text-red-600 hover:text-red-700 w-full px-3 py-2 rounded-md hover:bg-red-50 transition-colors"
                >
                    <LogOut size={20} />
                    <span className="text-sm font-medium">{t('nav.logout')}</span>
                </button>
            </div>
        </div>
    );
}
