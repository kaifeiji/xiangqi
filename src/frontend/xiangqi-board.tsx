import { useEffect, useRef } from 'react'
import { Xiangqiground } from 'xiangqiground'
import type { Api } from 'xiangqiground/api'
import type { Config } from 'xiangqiground/config'
import type { Key } from 'xiangqiground/types'
import { boardToFen, legalDests } from './game-utils'
import type { Game } from './types'

const START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'

interface XiangqiBoardProps {
  active: boolean
  game: Game | undefined
  orientation?: 'white' | 'black'
  lastMove: Key[] | undefined
  readOnly: boolean
  onMove: (origin: Key, destination: Key) => void
}

export function XiangqiBoard({ active, game, orientation = 'white', lastMove, readOnly, onMove }: XiangqiBoardProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null)
  const boardRef = useRef<Api | undefined>(undefined)

  useEffect(() => {
    if (!containerRef.current) {
      return
    }
    const config: Config = {
      fen: game ? boardToFen(game.board) : START_FEN,
      orientation,
      turnColor: game?.side_to_move === 'b' ? 'black' : 'white',
      check: game?.in_check ? (game.side_to_move === 'w' ? 'white' : 'black') : false,
      lastMove: game ? lastMove : undefined,
      // Xiangqiground 仅在初始化时为非 viewOnly 棋盘注册输入监听器。
      // 初始棋盘仍通过 selectable 和 movable 保持不可操作。
      viewOnly: false,
      highlight: { lastMove: Boolean(game), check: Boolean(game) },
      selectable: { enabled: Boolean(game?.is_human_turn) && !readOnly },
      movable: {
        free: false,
        color: game?.is_human_turn && !readOnly ? (game.side_to_move === 'w' ? 'white' : 'black') : undefined,
        dests: game ? legalDests(game.legal_moves) : undefined,
        showDests: Boolean(game),
        events: { after: onMove },
      },
      draggable: { enabled: false },
      animation: { enabled: true, duration: 180 },
    }
    if (boardRef.current) {
      boardRef.current.set(config)
    } else {
      boardRef.current = Xiangqiground(containerRef.current, config)
    }
  }, [game, lastMove, onMove])

  useEffect(() => {
    if (active) boardRef.current?.redrawAll()
  }, [active])

  useEffect(() => () => boardRef.current?.destroy(), [])

  return <div ref={containerRef} className="xiangqiground" aria-label="象棋棋盘" />
}