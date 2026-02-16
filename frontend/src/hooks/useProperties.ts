import { useState, useEffect, useCallback } from 'react';
import { client } from '../api/client';
import type { FeatureCollection } from '../types';

export function useProperties(bbox?: string, filters?: any) {
    const [data, setData] = useState<FeatureCollection | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchProperties = useCallback(async (currentBbox: string) => {
        try {
            setLoading(true);
            const params = new URLSearchParams();
            if (currentBbox) params.append('bbox', currentBbox);

            // Add other filters
            if (filters) {
                Object.keys(filters).forEach(key => {
                    if (filters[key] !== undefined && filters[key] !== '') {
                        params.append(key, String(filters[key]));
                    }
                });
            }

            const res = await client.get<FeatureCollection>(`/properties?${params.toString()}`);
            setData(res.data);
            setError(null);
        } catch (err) {
            console.error(err);
            setError('Error al cargar propiedades');
        } finally {
            setLoading(false);
        }
    }, [filters]);

    // Debounced fetch
    useEffect(() => {
        if (!bbox) return;

        const timeoutId = setTimeout(() => {
            fetchProperties(bbox);
        }, 500); // 500ms debounce

        return () => clearTimeout(timeoutId);
    }, [bbox, fetchProperties]);

    return { data, loading, error };
}
