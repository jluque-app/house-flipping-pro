import { useState, useEffect, useCallback } from 'react';
import { client } from '../api/client';
import type { FeatureCollection } from '../types';

export function useProperties(bbox?: string, filters?: any) {
    const [data, setData] = useState<FeatureCollection | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchProperties = useCallback(async (currentBbox: string) => {
        let isMounted = true;

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

            // Timeout implementation in 15 seconds
            const controller = new AbortController();
            // Use window.setTimeout explicitly to avoid NodeJS vs Browser type conflict if inferred wrong, 
            // though usually fine. Safe pattern.
            const timeoutId = setTimeout(() => controller.abort(), 15000);

            const res = await client.get<FeatureCollection>(`/properties?${params.toString()}`, {
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (isMounted) {
                setData(res.data);
                setError(null);
            }
        } catch (err: any) {
            console.error(err);
            if (isMounted) {
                if (err.name === 'AbortError' || err.code === 'ECONNABORTED') {
                    setError('Timeout: El servidor tarda demasiado en responder (Render cold start).');
                } else {
                    setError('Error al cargar propiedades');
                }
            }
        } finally {
            if (isMounted) setLoading(false);
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
