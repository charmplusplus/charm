// Coordinator wire protocol.
//
// Frame: [u32 totalLen][u32 type][payload of (totalLen - 4) bytes].
// All integers are network byte order (big-endian).
//
// Member record (used inside several payloads):
//   [u32 nodeId][u32 ucxAddrLen][ucxAddrLen bytes ucxAddr]
//
// Rank-side flow:
//   Initial rank:
//     -> REGISTER_INITIAL { nodeId, ucxAddr }                  (nodeId is the rank's
//                                                              PMI rank — coordinator
//                                                              uses it directly so both
//                                                              sides agree on numbering)
//     <- REGISTER_INITIAL_REPLY { nodeId, epoch, members[] }   (sent after coordinator
//                                                              has collected expected count)
//   Newcomer rank:
//     -> REGISTER_NEWCOMER { ucxAddr }
//     <- REGISTER_NEWCOMER_REPLY { epoch, members[] }          (immediate snapshot of
//                                                              current members so the
//                                                              newcomer can build
//                                                              speculative eps while
//                                                              waiting for COMMIT)
//     <- INTEGRATE { nodeId, epoch, members[] }                (pushed when committed;
//                                                              members[] are the FINAL
//                                                              compactly-renumbered set)
//   PE 0 driving a reconfig:
//     -> QUERY_PENDING
//     <- QUERY_PENDING_REPLY { count }
//     -> COMMIT { epoch, kills[], take }
//     <- COMMIT_REPLY { newNodeId, epoch, killedOldIds[], added[] }
//                                                              (delta against the
//                                                              caller's cached view;
//                                                              killedOldIds are in OLD
//                                                              numbering; added carry
//                                                              their final new nodeIds)
//   Surviving (non-initiator) members during reconfig:
//     RECONFIG is no longer pushed by the coordinator. Survivors receive the
//     delta from PE 0 via UCX chain-broadcast (see machine.C
//     UcxReconfigChainForward / UcxRecvReconfigBytes). UCX payload shape:
//     { u32 epoch, u32 killedCount, u32 killedOldIds[killedCount],
//       u32 addedCount, Member added[addedCount] }
//     Each receiver computes its own new nodeId locally from killedOldIds
//     (new = old - count_of_killed_ids_strictly_less_than_self).
//
//   Delta reconstruction (client-side, deterministic; matches server in
//   coordinator.cpp handleCommit):
//     1. From cached old members in OLD nodeId order, drop any whose nodeId is
//        in killedOldIds.
//     2. Renumber the survivors compactly to 0..S-1 (preserving relative order).
//     3. Append `added` members verbatim — their nodeIds are S..S+|added|-1.
//   Killed members during reconfig:
//     <- DIE { }                                                (pushed at COMMIT; receiver
//                                                              should exit)
//   Any rank during wireup:
//     -> BARRIER { epoch, nodeId }
//     <- BARRIER_REPLY                                          (after all alive ranks check in)

#pragma once

#include <cstdint>

namespace coord {

enum MsgType : uint32_t {
  REGISTER_INITIAL       = 1,
  REGISTER_INITIAL_REPLY = 2,
  REGISTER_NEWCOMER      = 3,
  INTEGRATE              = 4,
  QUERY_PENDING          = 5,
  QUERY_PENDING_REPLY    = 6,
  COMMIT                 = 7,
  COMMIT_REPLY           = 8,
  BARRIER                = 9,
  BARRIER_REPLY          = 10,
  RECONFIG               = 11,
  DIE                    = 12,
  REGISTER_NEWCOMER_REPLY = 13,
};

constexpr uint32_t HEADER_BYTES = 8;  // totalLen + type

}  // namespace coord
