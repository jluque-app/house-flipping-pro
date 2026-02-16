export function parseBBox(bbox?: string): [number, number, number, number] | null {
    if (!bbox) return null;
    const parts = bbox.split(",").map(s => s.trim());
    if (parts.length !== 4) return null;
    const nums = parts.map(x => Number(x));
    if (nums.some(n => Number.isNaN(n))) return null;

    const [minLon, minLat, maxLon, maxLat] = nums;
    if (minLon >= maxLon || minLat >= maxLat) return null;

    return [minLon, minLat, maxLon, maxLat];
}
