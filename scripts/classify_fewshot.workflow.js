export const meta = {
  name: 'corseacare-classify-fewshot',
  description: 'Few-shot classify candidate-crop montages against the reference guide; return per-box class.',
  phases: [{ title: 'Classify', detail: 'one agent per montage, few-shot with the reference guide' }],
}
const SCHEMA = { type: 'object', properties: {
  labels: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'number' }, cls: { type: 'string' }, plastic: { type: 'boolean' } },
    required: ['id', 'cls'] } } }, required: ['labels'] }
// args: { guide: "<abs path to reference guide png>", montages: ["<abs montage png>", ...] }
const guide = args.guide
const results = await parallel(args.montages.map(m => () => agent(
  `Reference guide image: ${guide}\nCandidate montage: ${m}\n` +
  `Each numbered cell is one marine particle. Using the reference guide, return for each cell id its ` +
  `class (one of: fragment, fibre, film, mousse, pellet, autre) and whether it is plastic. ` +
  `'autre' = organic/indeterminate. Read both images.`,
  { label: `classify:${m.split('/').pop()}`, phase: 'Classify', schema: SCHEMA })))
return { results: results.filter(Boolean) }
