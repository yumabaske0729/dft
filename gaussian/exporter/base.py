# DFT/exporter/base.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import csv
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable, Optional

import numpy as np


@dataclass
class ExportOptions:
    """
    出力ポリシー/書式の集中管理用オプション
    """
    # ディレクトリ／ファイル命名
    base_out_dir: str = "runs"        # 例: runs/<run_id>/output
    include_subdir: str = "output"    # 上の run_id 直下に作るサブフォルダ
    timestamp_format: str = "%Y%m%d_%H%M%S"
    run_id_fmt: str = "{ts}_{input_tag}_{formula}_{basis}_{method}"  # basis は後で upper()

    # ファイル名フォーマット
    energies_csv_name_fmt: str = "energies_{input_tag}.csv"   # 従来: final 行のみ
    summary_json_name_fmt: str = "summary_{input_tag}.json"
    matrices_npz_name_fmt: str = "matrices_{input_tag}.npz"

    # 追加: 同内容の JSON も同時出力（行列の実体を JSON 化）
    matrices_json_name_fmt: str = "matrices_{input_tag}.json"

    # 追加（既存のまま）
    scf_csv_name_fmt: str = "scf_{input_tag}.csv"             # 反復履歴
    details_json_name_fmt: str = "details_{input_tag}.json"   # 詳細一式（行列はメタ情報に縮約）

    # 表示書式
    energy_fmt: str = "{:.12f}"
    trace_fmt: str = "{:.8f}"

    # 保存対象の ON/OFF
    save_matrices: bool = True
    save_summary: bool = True
    save_energies: bool = True
    save_scf_history: bool = True
    save_details: bool = True

    # 追加: NPZ に加えて JSON も出すか（デフォルト ON）
    save_matrices_json: bool = True


class Exporter:
    """
    計算成果物（CSV/JSON/NPZ/JSON(行列)）の書き出しを担当するクラス。
    main.py からはこのクラスの API だけを呼ぶ。
    """
    def __init__(self, options: Optional[ExportOptions] = None):
        self.opt = options or ExportOptions()

    # ---- ネーミング/タグ系ユーティリティ ----
    @staticmethod
    def hill_formula(symbols: List[str]) -> str:
        """
        Hill 記法ライクな簡易式：
        - C があれば C, H を優先、その後はアルファベット順
        - C がなければ全てアルファベット順
        - 係数 1 は省略
        """
        from collections import Counter
        cnt = Counter(symbols)
        if "C" in cnt:
            order = ["C"] + (["H"] if "H" in cnt else []) + [k for k in sorted(cnt) if k not in ("C", "H")]
        else:
            order = sorted(cnt)
        parts = []
        for k in order:
            n = cnt[k]
            parts.append(k if n == 1 else f"{k}{n}")
        return "".join(parts)

    @staticmethod
    def safe_tag_from_path(path_str: str, maxlen: int = 64) -> str:
        """
        入力ファイルパスから拡張子なしのステム名を取り、Windowsでも安全なタグへサニタイズ。
        - 許可: A-Z a-z 0-9 . _ -
        - それ以外は '_' に置換
        - 先頭/末尾の . _ - はトリム
        - 長すぎる場合は切り詰め
        """
        base = os.path.basename(path_str)
        stem = os.path.splitext(base)[0]
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
        s = re.sub(r"^[._-]+", "", s)
        s = re.sub(r"[._-]+$", "", s)
        if not s:
            s = "input"
        if len(s) > maxlen:
            s = s[:maxlen]
        return s

    @staticmethod
    def build_spin_summed_density(
        C: np.ndarray, S: np.ndarray, mol, get_atomic_number: Callable[[str], int]
    ) -> Tuple[np.ndarray, int, float]:
        """
        最終 MO 係数 C から閉殻 RHF/DFT 用のスピン和密度行列 D を作り、Tr[DS] も返す。
        返り値: (D_final, n_occ, trace_DS)
        """
        nbf = C.shape[0]
        n_elec = sum(get_atomic_number(a.symbol) for a in mol.atoms) - mol.charge
        n_occ = int(n_elec // 2)
        D = np.zeros((nbf, nbf), dtype=np.float64)
        for m in range(n_occ):
            D += 2.0 * np.outer(C[:, m], C[:, m])
        trace_ds = float(np.einsum("mn,mn->", D, S))
        return D, n_occ, trace_ds

    def make_run_id_and_outdir(
        self, *, input_tag: str, formula: str, basis: str, method: str, timestamp: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        run_id と保存先 outdir (= <base_out_dir>/<run_id>/<include_subdir>) を返す
        """
        ts = timestamp or time.strftime(self.opt.timestamp_format)
        run_id = self.opt.run_id_fmt.format(
            ts=ts,
            input_tag=input_tag,
            formula=formula,
            basis=basis.upper(),
            method=method,
        )
        outdir = os.path.join(self.opt.base_out_dir, run_id, self.opt.include_subdir)
        return run_id, outdir

    # ---- 低レベル書き出し ----
    def _write_energies_csv(
        self, path: str, *, run_id: str, method: str, basis: str, input_tag: str,
        E_scf: float, trace_ds: float
    ) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "iter", "method", "basis", "input", "total_energy_hartree", "trace_DS"])
            w.writerow([
                run_id,
                "final",
                method,
                basis,
                input_tag,
                self.opt.energy_fmt.format(E_scf),
                self.opt.trace_fmt.format(trace_ds),
            ])

    def _write_scf_history_csv(self, path: str, *, run_id: str, history: List[Dict[str, Any]]) -> None:
        if not history:
            return
        # ヘッダ抽出（union of keys, 安定順序）
        keys = ["run_id", "iter", "E_total", "dE", "RMS_D", "TrDS", "E_Hcore", "E_2e", "E_JK",
                "Exc", "E_T", "E_Vne", "converged"]
        # 任意の追加キーも出てきたら後続に付与
        extra_keys = []
        for rec in history:
            for k in rec.keys():
                if k not in keys and k not in extra_keys:
                    extra_keys.append(k)
        header = keys + extra_keys
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for rec in history:
                row = [run_id]
                for k in header[1:]:
                    row.append(rec.get(k, ""))
                w.writerow(row)

    def _write_summary_json(
        self, path: str, *, run_id: str, method: str, basis: str, input_tag: str,
        E_scf: float, summary_extra: Optional[Dict[str, Any]]
    ) -> None:
        summary = {
            "run_id": run_id,
            "method": {"type": method, "basis": basis},
            "results": {"total_energy_hartree": float(E_scf)},
            "input": {"tag": input_tag},
        }
        if summary_extra:
            for k, v in summary_extra.items():
                if k == "results" and isinstance(v, dict):
                    summary["results"].update(v)
                else:
                    summary[k] = v
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _filter_arrays(arrs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in arrs.items():
            if v is None:
                continue
            a = np.asarray(v)
            out[k] = a
        return out

    def _write_matrices_npz(self, path: str, **arrays) -> None:
        # 可変長: 渡された配列をそのまま保存
        arrays = self._filter_arrays(arrays)
        if not arrays:
            return
        np.savez(path, **arrays)

    # 行列の JSON 版（NPZ と同じ内容を JSON 化）
    def _write_matrices_json(self, path: str, **arrays) -> None:
        arrays = self._filter_arrays(arrays)
        if not arrays:
            return

        def ndarray_payload(a: np.ndarray) -> Dict[str, Any]:
            return {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "data": a.tolist(),
            }

        blob = {
            "__format__": "DFT-matrices-json",
            "__version__": 1,
            "arrays": {k: ndarray_payload(v) for k, v in arrays.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)

    # ----- details を JSON 化できる形にサニタイズ -----
    @staticmethod
    def _jsonify(obj: Any) -> Any:
        """
        numpy のスカラー/配列を JSON 化可能な形に変換する再帰関数。
        - np.generic スカラー: Python スカラーへ
        - np.ndarray      : メタ情報 dict（実体は書かない）
        - dict/list/tuple : 再帰的に処理
        それ以外はそのまま返す。
        """
        # numpy scalar -> python scalar
        if isinstance(obj, np.generic):
            return obj.item()

        # numpy array -> meta only
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}

        # dict
        if isinstance(obj, dict):
            return {k: Exporter._jsonify(v) for k, v in obj.items()}

        # list / tuple
        if isinstance(obj, (list, tuple)):
            return [Exporter._jsonify(v) for v in obj]

        return obj

    def _details_sanitized(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        details のうち 'matrices' は配列メタ情報に縮約、
        それ以外は _jsonify で numpy スカラーをネイティブ化。
        """
        if not isinstance(details, dict):
            return details

        out = dict(details)
        mats = out.get("matrices", None)
        if isinstance(mats, dict):
            meta_only = {}
            for k, v in mats.items():
                if v is None:
                    meta_only[k] = None
                else:
                    a = np.asarray(v)
                    meta_only[k] = {"__ndarray__": True, "shape": list(a.shape), "dtype": str(a.dtype)}
            out["matrices"] = meta_only

        # 残りを再帰的に numpy -> python へ
        out = self._jsonify(out)
        return out

    # ---- ハイレベル API（main.py からはこれを呼ぶだけ）----
    def export_final(
        self,
        *,
        input_path: str,
        method: str,
        basis: str,
        mol,
        S: np.ndarray,
        T: np.ndarray,
        V: np.ndarray,
        C: np.ndarray,
        eps: np.ndarray,
        E_scf: float,
        get_atomic_number: Callable[[str], int],
        timestamp: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,  # 追加情報（matrices/履歴/診断を含む）
    ) -> Dict[str, str]:
        """
        最終結果を書き出し、生成されたファイルパスを dict で返す。
        """
        symbols = [a.symbol for a in mol.atoms]
        formula = self.hill_formula(symbols)
        input_tag = self.safe_tag_from_path(input_path)

        run_id, outdir = self.make_run_id_and_outdir(
            input_tag=input_tag, formula=formula, basis=basis, method=method, timestamp=timestamp
        )
        os.makedirs(outdir, exist_ok=True)

        # 密度と Tr[DS]
        D_final, n_occ, trace_ds = self.build_spin_summed_density(C, S, mol, get_atomic_number)

        # 追加サマリ情報（既存＋拡張）
        H_core = T + V
        summary_extra = {
            "system": {"formula": formula, "natoms": len(mol.atoms)},
            "results": {
                "total_energy_hartree": float(E_scf),
                "n_electrons": int(sum(get_atomic_number(a.symbol) for a in mol.atoms) - mol.charge),
                "n_basis": int(S.shape[0]),
                "n_occ_closed_shell": int(n_occ),
                "trace_DS": float(trace_ds),
            },
            "input": {
                "file": os.path.basename(input_path),
                "path": os.path.abspath(input_path),
                "tag": input_tag,
            },
        }
        # details からダイジェストをサマリへ反映
        if details:
            comps = details.get("components", {})
            diags = details.get("diagnostics", {})
            if comps:
                summary_extra.setdefault("results", {}).update(
                    {k: float(v) for k, v in comps.items() if isinstance(v, (int, float, np.generic))}
                )
            if diags:
                # numpy スカラーを素の型に
                summary_extra["diagnostics"] = self._jsonify(diags)

        # 各ファイルのパス
        energies_csv = os.path.join(outdir, self.opt.energies_csv_name_fmt.format(input_tag=input_tag))
        summary_json = os.path.join(outdir, self.opt.summary_json_name_fmt.format(input_tag=input_tag))
        matrices_npz = os.path.join(outdir, self.opt.matrices_npz_name_fmt.format(input_tag=input_tag))
        matrices_json = os.path.join(outdir, self.opt.matrices_json_name_fmt.format(input_tag=input_tag))  # 追加
        scf_csv = os.path.join(outdir, self.opt.scf_csv_name_fmt.format(input_tag=input_tag))
        details_json = os.path.join(outdir, self.opt.details_json_name_fmt.format(input_tag=input_tag))

        # 書き出し
        if self.opt.save_energies:
            self._write_energies_csv(
                energies_csv,
                run_id=run_id, method=method, basis=basis, input_tag=input_tag,
                E_scf=E_scf, trace_ds=trace_ds,
            )
        if self.opt.save_summary:
            self._write_summary_json(
                summary_json,
                run_id=run_id, method=method, basis=basis, input_tag=input_tag,
                E_scf=E_scf, summary_extra=summary_extra,
            )

        # NPZ: 既存＋拡張（details.matrices があれば併合）
        matrices_from_details = (details or {}).get("matrices", {}) if details else {}
        _arrays = dict(
            S=S, T=T, V=V, H_core=H_core, C=C, eps=eps, D_final=D_final,
            S_half=matrices_from_details.get("S_half"),
            S_evals=matrices_from_details.get("S_evals"),
            F_final=matrices_from_details.get("F_final"),
            J_final=matrices_from_details.get("J_final"),
            K_final=matrices_from_details.get("K_final"),
            Vxc_final=matrices_from_details.get("Vxc_final"),
        )
        if self.opt.save_matrices:
            self._write_matrices_npz(matrices_npz, **_arrays)

        # NPZ と同じ内容を JSON にも保存
        if self.opt.save_matrices and self.opt.save_matrices_json:
            self._write_matrices_json(matrices_json, **_arrays)

        # 反復履歴（CSV）
        if self.opt.save_scf_history and details and details.get("scf_history"):
            self._write_scf_history_csv(scf_csv, run_id=run_id, history=details["scf_history"])

        # 詳細 JSON（配列はメタ化、numpy スカラーは素の型に）
        if self.opt.save_details and details:
            details_sanitized = self._details_sanitized(details)
            # どのファイルに行列が保存されたか注記
            details_sanitized.setdefault("__matrices_files__", {
                "npz": os.path.basename(matrices_npz) if self.opt.save_matrices else "",
                "json": os.path.basename(matrices_json) if (self.opt.save_matrices and self.opt.save_matrices_json) else ""
            })
            block = {
                "run_id": run_id,
                "method": method,
                "basis": basis,
                "input_tag": input_tag,
                "details": details_sanitized,
            }
            with open(details_json, "w", encoding="utf-8") as f:
                json.dump(block, f, ensure_ascii=False, indent=2)

        # ログ表示
        print("[exporter] artifacts saved:")
        if self.opt.save_energies: print(f" - {energies_csv}")
        if self.opt.save_summary: print(f" - {summary_json}")
        if self.opt.save_matrices: print(f" - {matrices_npz}")
        if self.opt.save_matrices and self.opt.save_matrices_json: print(f" - {matrices_json}")
        if self.opt.save_scf_history and details and details.get("scf_history"): print(f" - {scf_csv}")
        if self.opt.save_details and details: print(f" - {details_json}")

        return {
            "run_id": run_id,
            "outdir": outdir,
            "energies_csv": energies_csv if self.opt.save_energies else "",
            "summary_json": summary_json if self.opt.save_summary else "",
            "matrices_npz": matrices_npz if self.opt.save_matrices else "",
            "matrices_json": matrices_json if (self.opt.save_matrices and self.opt.save_matrices_json) else "",
            "scf_csv": scf_csv if (self.opt.save_scf_history and details and details.get("scf_history")) else "",
            "details_json": details_json if (self.opt.save_details and details) else "",
        }
