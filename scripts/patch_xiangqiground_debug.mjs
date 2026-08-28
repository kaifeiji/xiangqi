import { readFile, writeFile } from 'node:fs/promises'

const files = [
  'node_modules/xiangqiground/src/fen.ts',
  'node_modules/xiangqiground/src/util.ts',
  'node_modules/xiangqiground/dist/fen.js',
  'node_modules/xiangqiground/dist/util.js',
]
const patterns = [
  /\s*console\.log\(col, row\);?/g,
  /\s*console\.log\(allKeys\);?/g,
  /\s*console\.log\(pos\);?/g,
]

for (const file of files) {
  let source
  try {
    source = await readFile(file, 'utf8')
  } catch {
    continue
  }
  const patched = patterns.reduce((value, pattern) => value.replace(pattern, ''), source)
  if (patched !== source) await writeFile(file, patched)
}
