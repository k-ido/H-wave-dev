.. highlight:: none

uhfk_to_mvmc.py — UHFk → mVMC PairProduct ブリッジ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tools/uhfk_to_mvmc.py`` は H-wave UHFk の SCF 結果を、mVMC の
``InOrbital`` / ``InOrbitalAntiParallel`` 初期波動関数ファイル
(``zqp_orbital_uhfk.dat``) に変換するスクリプトである。これにより
mVMC の PairProduct 状態を H-wave UHF の Slater 行列で初期化できる。

スコープ (v3):

- 単軌道 (``norb_orig = 1``)。副格子フォールドは ``SubShape`` が各方向で
  ``CellShape`` を割り切る任意の値に対応。``SubShape`` 省略時は
  ``CellShape`` にフォールバック (``uhfk.py:_init_lattice`` と一致)。
- **Sz-fixed UHF (2Sz = 0)** → AntiParallel path (v1/v2/v2.1 と同じ
  挙動、``orbitalidx.def`` の 3 or 4 列形式を消費)。
- **Sz-fixed 2Sz ≠ 0 (A ケース) と Sz-free 非 mixed (B ケース)** →
  General path (v3、``orbitalidx_general.def`` の 6 列形式を消費)。
  CLI は ``(is_antiparallel_metadata, orbitalidx_format)`` の組で
  自動 dispatch する。ユーザは StdFace で ``orbitalidx_general.def``
  を生成する必要がある (A ケースでは ``stan.in`` の ``2Sz`` を非 0 に
  設定する)。
- スピン不均衡 Slater については、canonical ``(k, partner(k))`` ブロック
  での same-spin excess pair 発行により **同スピン pair 成分**
  (``F[up, up]``, ``F[down, down]``) をサポートする。
- Sz-free で mixed block を含む状態 (``column_spin = -1``、SOC /
  spin-orbital mode) は **v3 スコープ外**、v3.1 で扱う。
- PBC・APBC とも決定論的にサポート。ブリッジは、H-wave の負号ゲージ
  APBC 変換 (``tilde_c_r = exp(-i theta r / L_phys) c_r``) と、
  ``ifftn(..., norm='forward')`` に対応する tilde 側の正号 Bloch
  Fourier (``c_R = (1/sqrt(N_folded)) sum_k c_k exp(+i k R)``) から
  導かれる **正号 Bloch 振幅規約** を採用する。
  ``SubShape = [1, 1, 1]`` では (k, -k) 時間反転対称和で plane wave 因子
  が対称化されるため密度は符号選択に依存しないが、``SubShape > [1, 1, 1]``
  では折り畳み固有ベクトルの副格子 envelope が非自明になるため、H-wave
  ``greenone.dat`` と element-wise 一致するのは本規約のみ (SubShape=[2,1,1]
  APBC L=8 fixture で 1e-14 に確認済み)。
- AntiParallel path: (k, -k) 時間反転対、``(k_row, local_band)`` 上で
  pair-closure。General path (v3): canonical ``(k, partner(k))``
  ブロック内で cross + same-spin excess pair を発行。§3.2 の
  pair-closure 条件 (canonical block の same-spin excess 不整合、
  self-pair の奇数 excess など) を満たさない占有は
  ``validate_general_prerequisites`` で明示的に拒絶する。
- T=0 への Slater 投影。Fermi 準位付近に分数占有が残る有限温度 SCF は拒絶
  (より小さい ``T`` で再計算するよう促す)
- ``params[idx]`` に微小な一様乱数(default 振幅 ``1e-8``、
  ``--epsilon-noise`` で変更可)を加算して rank-deficient な F に対する
  mVMC の Pfaffian Slater 評価の特異性を回避する。``ComplexType 1``
  では実部・虚部ともに、``ComplexType 0`` では実部のみに乱数を加える。
  mVMC 本体の ComplexUHF と同じ手法
  (``mVMC-1.4.0/src/ComplexUHF/output.c:274``)。

スコープ (v3.1)
^^^^^^^^^^^^^^^

- **Sz-free mixed block (SOC、``enable_spin_orbital = true``)** →
  General-SOC path。``orbitalidx_general.def`` (6 列) を消費し、Sz 非保存の
  ``F[up_i, down_j]`` / ``F[down_i, up_j]`` を出力する。Zeeman、Rashba、
  Dresselhaus、一般 σ_x/σ_y 型 1-body coupling をサポート。
- ``eigen.npz["twist_offset"]`` を ``input.toml`` の canonical
  ``BoundaryCondition`` と照合し、stale 入力/eigen のペアを排除する。
- SOC 下では bridge が H-wave の ``Transfer.dat`` から ``trans.def`` も
  出力する (下記参照)。mVMC の ``vmcdry.out`` は Rashba の s!=t 項を
  保存できない。

スコープ (v3.2)
^^^^^^^^^^^^^^^

- **SOC + APBC** を end-to-end でサポート。``trans.def`` 出力に
  ``BoundaryCondition`` 由来の ``boundary_theta`` を透過させる:
  APBC 方向で境界を跨ぐ行 (``(R_x, R_y, R_z)`` が境界越え) は物理的な
  wrap-phase (正の ``R`` 越えで ``exp(i theta_d)``、負の ``R`` 越えで
  ``exp(-i theta_d)``、APBC 方向は ``theta_d = pi``) を掛ける。
  ベースの SOC 符号規約 (下の「符号規約」参照) と同様、この wrap-phase
  規約も **empirically pinned** であり、
  ``tests/validation/uhfk_mvmc_pairproduct/case_soc_rashba_2d_nosub_apbc``
  で E2E 検証済み (mVMC ⟨H⟩ が H-wave UHF から 0.03% 以内)。H-wave 内部
  規約から first-principles で導出することは UHFk path では clean には
  通らない — 詳細は ``tools/_uhfk_to_mvmc/trans_emit.py`` の module
  docstring を参照。
- **SOC + SubShape > [1, 1, 1]** は v3.4 で延期されていた (Codex v3.4
  Rev.2 finding: dual-A 密度 gate では ship される A/F を検証できず、
  reference A が ``sub_offset`` を落とした別物であったため)。v3.5 では
  この経路を、SHIPPING A を直接検証する gauge-lifted 単一 A 密度チェック
  で復活させた — 下の「Scope (v3.5)」を参照。
- **SOC + APBC + SubShape > [1, 1, 1]** の三重組合せは v3.5 でも引き
  続き延期する (Rev.1 finding 2: E2E fixture 未整備、SOC + APBC と
  SOC + SubShape を合成した phase 経路は独立検証されていない)。
  ``enable_spin_orbital = true`` と antiperiodic ``BoundaryCondition``
  と ``SubShape > [1, 1, 1]`` が同時に指定されると、CLI は dispatch
  前に fail-fast する。
- v3.1 SOC ``trans.def`` の符号規約は引き続き **empirically pinned**。
  以前 ``epsilon_k`` swap と ``H = -sum trans`` の合成から導出すると記載
  していたが、``sc.py`` の ``epsilon_k[orb2, orb1]`` swap は UHFk path
  (``uhfk.py``) では行われないため、この swap 由来の解析的導出は本
  ブリッジには当てはまらないことが判明した。ship される規約は
  ``case_soc_rashba_2d_nosub`` に対する ComplexUHF 検証で 4.4e-8%
  で再検証されており、詳細は ``tools/_uhfk_to_mvmc/trans_emit.py`` の
  module docstring (empirical basis と、放棄された derivation の
  経緯) を参照。

スコープ (v3.5)
^^^^^^^^^^^^^^^

- **SOC + SubShape > [1, 1, 1]** は v3.5 で、gauge-lifted 密度チェック
  ``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
  (element-wise 1e-10) を経由してサポートされた。密度チェックは H-wave
  の ``green_sublattice`` (folded-Bloch basis) を ``gauge_lift`` で物理
  基底に持ち上げ、SHIPPING ``conj(A) @ A.T`` (= mVMC に emit する A
  そのもの) と element-wise 照合する。これにより v3.4 の dual-A
  gap を塞ぐ。gauge 変換は shipping A の位相
  ``exp(-i k · (folded_cell + sub_offset))`` を ``green_sublattice``
  の folded-Bloch 保存形式に接続する; 導出は
  ``docs/superpowers/specs/2026-07-05-uhfk-mvmc-pairproduct-general-v35-design.md``
  §2-3 を参照 (``docs/superpowers/`` は gitignored で、リリースには
  spec は同梱されない)。v3.4 の ``_soc_reference_convention``
  escape hatch は削除され、コードベースには shipping 規約の A のみが
  残る。
- **SOC + APBC + SubShape > [1, 1, 1]** の三重組合せは引き続き
  reject される (合成 phase 経路が未検証、E2E fixture 無し — 上の
  「Scope (v3.2)」も参照)。
- SOC + SubShape > [1, 1, 1] で必須の CLI フラグ: ``--transfer``
  (H-wave ``Transfer.dat`` の読み込み)、``--emit-trans`` (Rashba の
  spin-off-diagonal を保持した mVMC ``trans.def`` の出力)、
  ``--emit-orbitalidx`` (StdFace の class merging を bypass。折り
  畳み格子下では Sz 非保存クラスを表現できない — v3.2 spec §1 の
  rationale を参照)。
- Empirical E2E: ``case_soc_rashba_2d_sub`` で mVMC ⟨H⟩ が
  H-wave ``Energy_Total`` と 0.22% 一致 (delta -0.055 / 25.10)、
  gauge-lifted 密度 gate は 1e-10 で clean。

スコープ (v3.6)
^^^^^^^^^^^^^^^

- **SOC + 単方向 APBC + SubShape > [1, 1, 1]** を v3.6 でサポートする。
  ``build_slater_orbitals`` が sub_offset gauge
  ``exp(-i k_folded · (folded_cell + sub_offset))`` と APBC twist
  ``exp(-i theta · r_phys / L_phys)`` を合成し、shipping A に両位相が
  一貫して乗る。``tools/_uhfk_to_mvmc/density_check.py`` の
  ``gauge_lift`` は ``boundary_theta`` を受け取り、同じ合成変換で
  ``green_sublattice`` を物理基底に持ち上げる。shipping 密度 gate
  ``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
  は v3.6 の shipping fixture ``case_soc_rashba_2d_sub_apbc``
  (``CellShape = [6, 4, 1]``, ``SubShape = [2, 2, 1]``,
  ``BoundaryCondition = ["antiperiodic", "periodic", "periodic"]``,
  ``enable_spin_orbital = true``, Rashba ``alpha = 0.5``, ``U = 2``,
  ``Ncond = 8``) 上で 1e-10 を通過する。

- **多方向 APBC (n_apbc_dirs >= 2) + SOC + SubShape > [1, 1, 1]**
  は v3.6 では pre-dispatch で reject していた。多方向 APBC +
  SubShape > 1 の合成 twist gauge が E2E fixture 上で未検証であり、
  誤った ``trans.def`` / shipping A を silent に生成することを避ける
  ための fail-fast であった。**v3.7 で解消**: v3.6 の reject
  メッセージと ``n_apbc_dirs > 1`` 述語は現在の tree に存在しない。
  v3.7 は検証済みの多方向 fixture を 4 つ同梱し、両者を後述の
  スコープ (v3.7) の allowlist 述語で置き換えている。

- **v3.6 七 gate 契約** (fresh workspace ``case_soc_rashba_2d_sub_apbc``
  で全 gate が PASS)::

    1. G0-writer-check: emitted-F の rank-lift ノイズ off 経路が
       aggregate 済 (mapping, params) と 1e-10 で一致
    2. G1: shipping ``build_slater_orbitals`` 密度が gauge_lift-lifted
       ``green_sublattice`` と 1e-10 で一致
    3. G2a-emitted-F: emitted-F projector 密度が ComplexUHF 一体
       Green と 1e-6 で一致
    4. G2a-in-memory-A: in-memory shipping A 密度が ComplexUHF と
       1e-6 で一致
    5. G2b: gauge-lifted ``green_sublattice`` が ComplexUHF と 1e-6 で一致
    6. G3: mVMC ⟨H⟩ と H-wave ``Energy_Total`` の相対 delta ≤ 1 %
    7. G4: ``composite_element.json`` の合成 ``(i_c, s_c, j_c, t_c)``
       が現行 SCF 上で維持され、M-gauge-1..5 + M-ship-1..5 の 10
       mutation 全てが ``T_M = max(1e-5, 0.10 * |G_base|)`` (spec §4.4)
       を超過

  gate は
  ``tests/validation/uhfk_mvmc_pairproduct/run.sh case_soc_rashba_2d_sub_apbc``
  から実行される。各 gate は
  ``^GATE_NAME PASS mode=... `` 形式の anchored PASS 行を出力し、
  ``awk 'index($0, p) == 1'`` で検証する。

- **v3.6 hardening pass (Codex adversarial-review 2026-07-12)**。
  Phase 6 の初回 seven-gate PASS 後、4 ラウンドの adversarial review で
  一連の findings が浮上し、以下のように修正した:

  * **G1 / G2a / G2b 実比較**: 初回の dispatcher は ``max_abs_delta=0.0``
    で PASS を印字する stub で、実際の numeric 比較は行っていなかった。
    ``compare.py`` を全面書き換えし、workspace の H-wave outputs を読み
    込んで ``build_slater_orbitals`` で shipping A を組み、
    ``gauge_lift`` (workspace の ``boundary_theta``) で
    ``green_sublattice`` を持ち上げ、strict-parse された ComplexUHF
    ``zvo_UHF_cisajs.dat`` と比較する形にした。ファイル欠損 / 打切り /
    形式不正 / 重複 / 範囲外 / 非有限値は新設 ``ComplexUHFParseError``
    で fail closed。

  * **ComplexUHF Cross-solver seeding**: Rashba + APBC 下で H-wave の
    broken-symmetry 極小と ComplexUHF の random-init default は異なり、
    2 つの独立 SCF が別の valid 極小に落ちて 4.76e-2 の要素毎密度差が
    生じた。修正: ``scripts/seed_complexuhf_from_hwave.py`` が
    H-wave の shipping A 密度を ComplexUHF ``initial.def`` (
    ``IgnoreLinesInDef=5`` 規約に沿った 5 行 header) に書き出し、微小な
    Hermitian ``perturb-scale`` を付与することで ComplexUHF を実際に
    反復させるが、H-wave の basin を離脱しない範囲に留める。
    本節が当初記載していた値 (``perturb-scale`` 1e-6、9 SCF steps、
    ``Energy_Total = -25.3717166``) は既に古い。1e-6 は G2 の許容値と
    同値であり、seed をそのまま返すソルバでも G2 を通過できてしまった。
    現在このフィクスチャは ``flag_fock = true`` のもと
    ``perturb-scale`` 1e-3 で走り、ComplexUHF は 70 SCF steps で
    ``Energy_Total = -25.390269883203`` に収束する。旧来の
    「SCF が 1 step 以上走ったこと」の表明を置き換えた収縮要件に
    ついては、上記スコープ (v3.7) を参照。

  * **Snapshot workspace rejection**: snapshot guard が ``tests/data``
    をプロセス CWD 基準で resolve していたため、リポジトリ外から起動
    すると check が silent に bypass されていた。修正: guard module
    自身の場所 (``Path(__file__).resolve().parents[3]``) に anchor し、
    anchored root が存在しなければ fail closed。

  * **G4 shadow-copy drift**: topology guard の
    ``_build_A_ship_mutated`` が SOC branch を inline に再実装しつつ、
    baseline を canonical kernel と比較していなかった。
    ``build_slater_orbitals`` に drift が入っても G4 が見逃す可能性が
    あった。修正: mutation matrix 実行前に canonical kernel との 1e-10
    等値 assert を必須化。

  * **Loader-injection env sanitization**: ``run.sh`` が caller 側の
    ``LD_LIBRARY_PATH`` / ``LD_PRELOAD`` / ``LD_AUDIT`` を子プロセス
    (``vmcdry.out`` / ``vmc.out`` / ``UHF``) にそのまま継承していた。
    修正: (a) 上記変数 + ``BASH_ENV`` / ``ENV`` を top-level で
    ``unset``, (b) ``trap - DEBUG ERR RETURN EXIT`` で継承 trap を
    クリア, (c) ``run_native()`` wrapper が
    ``env -u LD_PRELOAD -u LD_AUDIT [-u LD_LIBRARY_PATH|
    LD_LIBRARY_PATH=<validated>]`` を用いて各 native solver 呼び出し
    毎に per-command 消毒, (d) ``MVMC_LD_LIBRARY_PATH`` を絶対パス /
    現在の user 所有 / non-world-writable / 単一ディレクトリで validate。

  各 findings に対応する regression tests を追加: loader-env
  sanitization に 5 tests (`tests/test_run_sh_loader_env_sanitize.py`)、
  ComplexUHF strict parser に 6 tests
  (`tests/test_uhfk_mvmc_pairproduct_compare_wiring.py`)、
  CWD-independent snapshot guard に 3 tests
  (`tests/test_snapshot_rejection_guard_v36.py`)。

スコープ (v3.7)
^^^^^^^^^^^^^^^

- **SOC + 多方向 APBC + SubShape > [1, 1, 1]** を v3.7 でサポートする。
  対象は ``CellShape = [4, 4, 4]`` / ``SubShape = [2, 2, 2]``
  (folded BZ ``[2, 2, 2]``、物理サイト 64、スピン軌道次元 128) 上の
  xy / yz / xz / xyz の 4 つの活性方向マスク。shipping fixture は
  4 つで、いずれも xy 面内 Rashba ``alpha = 0.5``、スピン対角な z
  方向ホッピング ``t_z = -1``、係数 ``0.3 + 0.4j`` の一般的な複素 z
  方向スピン混合ホッピング、``U = 2`` を持つ::

      case_soc_rashba_3d_sub_apbc_xy    AP  AP  P    Ncond = 20
      case_soc_rashba_3d_sub_apbc_xz    AP  P   AP   Ncond = 20
      case_soc_rashba_3d_sub_apbc_yz    P   AP  AP   Ncond = 24
      case_soc_rashba_3d_sub_apbc_xyz   AP  AP  AP   Ncond = 12

  z 方向スピン混合ブロックは Rashba SOC ではない。
  ``H_z(k_z) = (0.6 cos(k_z) - 0.8 sin(k_z)) sigma_x`` となり、偶関数成分
  ``0.6 cos(k_z)`` がスピン 1/2 の時間反転対称性を破るためである。
  ホッピング全体は Hermitian のままであるため、これらの fixture は
  写像の検証に有効であり、一般的な複素ホッピングも検査できる。
  時間反転対称な 3D SOC fixture は v3.8 の follow-up とする。

  ``Ncond`` は fixture 間で一様ではない。各 fixture は、自身の収束
  スペクトル上で ``>= 5e-2`` の HOMO-LUMO gap と ``build_pair_list``
  の partner-balance 不変条件 ``n_occ(k) == n_occ(partner(k))`` を
  ともに満たす充填に pin されている。スカラーの gap 判定だけでは
  不十分である。gap は canonical/partner 対の 2 行に占有状態が
  どう分配されるかについて何も語らないためである。候補ごとの
  完全なスキャン結果は各 fixture の ``README.md`` に記録している。

- **``flag_fock = true`` が必須** であり、任意設定ではない。検証用
  バイナリ ``ComplexUHF`` はオンサイト交換項をコンパイル時に固定
  しており (``src/ComplexUHF/include/Def.h`` の ``#define Fock 1``)、
  実行時スイッチを持たない。したがって ``flag_fock = false`` の
  H-wave 実行は、比較対象のソルバとは異なる平均場汎関数を最小化
  することになる。この不整合は、収束後のオンサイト横スピン密度が
  無視できる場合は顕在化しない (v3.6 fixture では 8.6e-17)。しかし
  v3.7 の z-SOC ブロックはこの密度を 2.7e-2 まで押し上げるため、
  Hartree のみの H-wave 解は ComplexUHF のいかなる不動点からも
  3.1e-3 離れる。これは G2a 許容値の 3000 倍であり、seed の選び方
  では到達できない。

- **``trans.def`` のスピン非対角写像**。Transfer.dat のエントリ
  ``(R, s, t, v)`` に対し、bridge はスピン端点を入れ替え、係数を
  共役かつ符号反転して出力する::

      K[i, t; i+R, s]     = conj(v)
      trans[i, t; i+R, s] = -conj(v)

  サイト端点は変わらない。実数のスピン対角成分では入れ替えが
  no-op となり、規則は ``trans = -v`` に帰着する。v3.6 の x/y
  Rashba 行列は固定 ``R`` において ``v[t,s] = -conj(v[s,t])`` を
  満たし、これは本規則と従来の非対角規則 ``trans = +v`` が同一の
  行列を出力する条件そのものであるため、v3.6 の結果は影響を
  受けない。v3.7 の z-SOC ブロックはスピン対称かつ実部・虚部を
  ともに持ち、この条件を満たさない。これが一般規則を必要とした
  理由である。境界の wrap 位相は共役の後に適用する。同梱の
  fixture はすべて ``theta`` 成分が ``{0, pi}`` であり、この位相は
  実数である。一般の複素 twist は対象外であり未検証である。

- **v3.7 allowlist**。SOC かつ APBC 方向を持つ ``SubShape > [1, 1, 1]``
  の組み合わせは、``tools/_uhfk_to_mvmc/allowlist_predicate.py`` の
  明示的な allowlist で検査する。CLI と静的カバレッジ checker が
  この述語を共有するため、両者が乖離することはない::

      _V37_ALLOWED_APBC_MASKS = {(1,1,0), (1,0,1), (0,1,1), (1,1,1)}
      _V37_LATTICE            = ((2,2,2), (4,4,4))   # sub_shape, cell_shape
      _V36_ALLOWED_APBC_MASKS = {(1,0,0), (0,1,0), (0,0,1)}
      _V36_LATTICE            = ((2,2,1), (6,4,1))

  非 SOC、``SubShape = [1, 1, 1]``、SOC かつ全方向周期境界の場合は
  early-return でサポート扱いとなる。それ以外で allowlist に無いもの
  は dispatch 前に以下のメッセージで reject される::

      ERROR: SOC + APBC + SubShape combination not in the v3.7
      allowlist. Supported active-direction masks + shapes: (a) v3.6
      single-dir APBC on CellShape=[6,4,1]/SubShape=[2,2,1]; (b) v3.7
      xy/xz/yz/xyz APBC on CellShape=[4,4,4]/SubShape=[2,2,2]. Others
      are deferred; add a new fixture + gate validation before
      expanding the allowlist.

- **v3.7 七 gate 契約**: スコープ (v3.6) に挙げた 7 つの gate を
  4 fixture それぞれで実行し、計 28 個の anchored PASS 記録を得る。
  fresh workspace における 4 fixture 中の最悪値::

      G0-writer-check    4.16e-17   tol 1e-10
      G1                 3.41e-13   tol 1e-10
      G2a-emitted-F      1.59e-07   tol 1e-06
      G2a-in-memory-A    1.55e-07   tol 1e-06
      G2b                1.55e-07   tol 1e-06
      G3                 4.49e-04   tol 1e-02
      G4                 0.00e+00   tol 2e-01

- **G2 は一致するだけでなく収縮しなければならない**。ComplexUHF は
  H-wave の収束密度から seed される。これは両ソルバを同じ対称性の
  破れた極小に収めるためだが、比較を循環させる危険がある。seed が
  既に許容値を満たしていれば、それをそのまま返すソルバでも通過して
  しまうためである。そこで各 G2 は、seed 時点で基準から少なくとも
  ``10 * tol`` 離れていることと、収束後に ``tol`` の内側に入ることの
  両方を要求し、双方と収縮率を記録する::

      G2b PASS ... initial_delta=5.037430e-04 final_delta=2.705664e-08
                   contraction_ratio=5.371120e-05

  この要件は G2 を走らせる全 fixture に例外なく適用される。また
  比較の前に非有限値を拒否する。NaN の差分は両方の境界比較を False に
  してしまい、そのままでは素通りするためである。

  これが可能なのは、v3.6 と v3.7 の shipping fixture がいずれも
  ``flag_fock = true`` になったからである。使用可能な seed の窓が
  そもそも存在するかは汎関数に依存する。H-wave と ComplexUHF が
  一致していれば基底は広く、v3.7 の格子では 1e-2 の seed でも戻るため
  ``1e-3`` は許容値の約 500 倍外側から余裕をもって始められる。
  一致していない場合は窓が皆無になりうる。``flag_fock = false``
  時代の v3.6 fixture では、H-wave の密度は ComplexUHF の写像に対して
  定常ではあるが**反発的**な点であり、5e-6 の seed でも 4.761e-2 まで
  脱出した。その変位はすべて Fock 項が作用するオンサイト横スピン成分に
  沿っている。基底半径は G2 許容値の 2.4 倍しかなく、「許容値の内側」と
  「基底の外側」の間に隙間が無かった。flag_fock を統一したことで、
  この fixture の G2 は定常性の確認から真の収束検証に変わり、
  3.888e-04 から 2.437e-08 へ収縮するようになった。

  上表の v3.7 の G2 値は、収縮要件の導入前に本書が記載していた値
  (約 3e-8) より大きい。小さかったのは seed をほぼ答えの位置に
  置いていたことによる見かけ上のもので、約 1.6e-7 が xyz fixture に
  おける ComplexUHF の正直な収束一致である。xyz は姉妹 fixture
  (16〜17 ステップ) より収束が遅く 91 ステップを要する。

  G4 はさらに、各 fixture に同梱した ``composite_element.json`` に
  対して方向ごと 30 エントリの mutation matrix を再検証する。schema
  は常に 30 エントリだが、gate するのは active axis のエントリだけで
  ある。xy/xz/yz fixture はそれぞれ、正の threshold を持つ 20 evaluation
  と inactive axis の zero-threshold 10 エントリを持つ。xyz fixture は
  30 evaluation すべてが正の threshold を持つ。したがって 9 active
  axis に対し、4 fixture 全体で 90 個の相異なる policy-gated mutation
  を評価する。従来の manifest では M-4 が sub_offset の符号を反転して
  いたが、これは ``L_folded = [2, 2, 2]`` 上で厳密な no-op であり、
  active axis の 18 エントリ (4 + 4 + 4 + 6) に暗黙に zero threshold
  を割り当て、実効 evaluation 数を 72 にしていた。その後の暫定修正は
  方向別 M-ship-4 と M-ship-5 の両方で sub_offset を省略したため、正の
  threshold は 90 個でも相異なる evaluation は 81 個にとどまった。
  M-gauge-4 は指定 axis の sub_offset を省略し、M-ship-4 は寄与を半分に
  し、M-ship-5 は省略する。v3.6 の 10 エントリ whole-vector schema の
  符号反転 semantics は変更していない。manifest producer は構造的退化、
  非有限数、threshold policy の不一致、sub-threshold self-check のいずれ
  でも fail closed する。runtime guard も threshold policy を独立に再計算
  する。4 つをまとめて
  実行するには
  ``tests/validation/uhfk_mvmc_pairproduct/run.sh --all-v37``、
  個別に実行するには case 名を引数に渡す。

ワークフロー
^^^^^^^^^^^^

1. StdFace で mVMC 入力一式 (``orbitalidx.def`` 等) を生成する。
   ``CalcMode = 2`` と、副格子並進対称性に応じた ``Lsub`` を設定。
   APBC は ``phase0 = 180.0`` を指定。
2. H-wave UHFk SCF を実行し、``eigen.npz`` に加えて新規 ``occupation.npz``
   を ``[file.output]`` で要求する::

       [file.output]
         path_to_output = "output"
         eigen          = "eigen.npz"
         green          = "green.npz"
         occupation     = "occupation.npz"
         onebodyg       = "greenone.dat"

3. ブリッジを実行::

       python tools/uhfk_to_mvmc.py \
           --input        input.toml \
           --eigen        output/eigen.npz \
           --occupation   output/occupation.npz \
           --geometry     geometry_uhf.dat \
           --orbitalidx   mvmc_inputs/orbitalidx.def \
           --output       mvmc_inputs/zqp_orbital_uhfk.dat \
           --check-density \
           --onebodyg-uhf output/greenone.dat

4. mVMC の ``namelist.def`` に ``InOrbital zqp_orbital_uhfk.dat``
   (または ``InOrbitalAntiParallel zqp_orbital_uhfk.dat``、v3 General
   path なら ``InOrbitalGeneral zqp_orbital_uhfk.dat``) を追加すれば、
   mVMC が PairProduct パラメータを本ファイルで初期化する。

Dispatch (v3)
^^^^^^^^^^^^^

CLI は ``--orbitalidx`` を先に parse し、
``(is_antiparallel_metadata, orbitalidx_format)`` の組で経路を選択する:

- ``(True, antiparallel)`` — v2.1 AntiParallel path (v1/v2/v2.1 の
  挙動そのまま)。
- ``(True, general)`` — forced-General 分岐。占有集合が v2.1 の
  ``(k_row, local_band)`` pair-closure を満たしていれば
  ``F[up, down]`` は 1e-12 で v2.1 の F を再現する。満たしていない
  (spin canting 由来の up-up excess 等) 場合は WARNING を出しつつ、
  出力は正しい ``InOrbitalGeneral`` state となる (ただし v2.1 経路
  では再現不可)。
- ``(False, general)`` — v3 General path (A + B スコープ)。
- ``(False, antiparallel)`` — 拒絶。StdFace を非 0 の ``2Sz`` (A) や
  Zeeman 駆動 Sz-free (B) 設定で回して ``orbitalidx_general.def`` を
  再生成すること。

``is_antiparallel_metadata`` は ``input.toml`` の ``2Sz`` が明示的に 0、
``N_up == N_down``、``column_spin ∈ {0, 1}``、``column_mu_group`` の
unique 数が 2、column_spin と mu_group が bijective であることを **全て**
満たしたときのみ True になる。

Dispatch (v3.1、6-case)
^^^^^^^^^^^^^^^^^^^^^^^

CLI は ``input.toml`` を parse した後
``is_soc_mode = toml_param.get("enable_spin_orbital", False)`` を計算し、
``(is_antiparallel_metadata, orbitalidx_format, is_soc_mode)`` の組で
分岐する:

- ``(True, antiparallel, False)`` — v2.1 AntiParallel (変更なし)。
- ``(True, general, False)`` — v3 forced-General。
- ``(False, general, False)`` — v3 General (A/B)。
- ``(False, antiparallel, False)`` — 拒絶。
- ``(\*, general, True)`` — **v3.1 General-SOC**。
- ``(\*, antiparallel, True)`` — 拒絶 (SOC は 6 列必須)。

BoundaryCondition の契約 (v3.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

bridge は ``BoundaryCondition`` を H-wave の共有 helper
``normalize_boundary_condition`` に委譲して正規化する。受理する形式:

- PBC: ``"p"``, ``"periodic"`` (case-insensitive、whitespace-stripped)
- APBC: ``"ap"``, ``"antiperiodic"`` (同上)

これ以外の文字列は dispatch 前に ``ValueError``。raw fallback は存在しない。
key を省略すると all-PBC を default とする。

Bridge trans.def 出力 (v3.1 SOC 限定)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SOC 下では mVMC の ``vmcdry.out`` が ``StdFace_Hopping`` で ``trans.def``
を作るが、spin-diagonal のみで Rashba の s != t 項が落ちる。この gap を
埋めるため、bridge が H-wave の ``Transfer.dat`` (Wannier90 風形式、
``iWan = 2 * a_phys + spin + 1``) を読み、mVMC の ``trans.def`` を
``(i, s, j, t, re, im)`` 形式で書き出す。Rashba の spin-off-diagonal を
保持する。

写像規約:

``Transfer.dat`` のエントリ ``(R, s, t, v)`` に対し、emitter はスピン
端点を入れ替えて共役を取る:

- ``K[i, t; i+R, s] = conj(v)``
- ``trans[i, t; i+R, s] = -conj(v)``

サイト端点は変わらない。実数のスピン対角成分では入れ替えが no-op と
なり、規則は ``trans = -v`` に帰着する (vmcdry の flip と一致)。
mVMC の規約は ``H = -Σ trans c†c`` である。

これは旧規則 (``s == t`` で ``-val``、``s != t`` で ``+val``) を置き換える。
旧規則は導出されたものではなく経験的に固定されたもので、等価ではなく
特殊ケースである。v3.6 の x/y Rashba 行列は固定 ``R`` において
``v[t,s] = -conj(v[s,t])`` を満たし、これは両規則が同一の行列を出力する
条件そのものである。v3.7 の z-SOC ブロックはスピン対称かつ実部・虚部を
ともに持ちこの条件を満たさない。実際、H-wave の bare ``K`` の再現精度は
旧規則が ``6.0e-01`` (``1e-10`` 超過 256 エントリ)、一般規則が
``1.1e-12`` (超過なし) であった。

v3.6 との互換性は行列等価であって byte 一致ではない。非対角行や一部の
符号付きゼロで出力テキストは異なるが、組み上がる Hamiltonian は同一で、
v3.6 の seven-gate E2E は変わらず通過する。導出と検証値は
``tools/_uhfk_to_mvmc/trans_emit.py`` の module docstring を、境界位相の
適用範囲は上記スコープ (v3.7) を参照。

新規 CLI フラグ (SOC 限定): ``--transfer <path>`` と ``--emit-trans <path>``。

v3.5 のスコープ外
^^^^^^^^^^^^^^^^^

- SOC + APBC + ``SubShape > [1, 1, 1]`` の三重組合せは v3.5 以降へ
  延期 (Rev.1 finding 2: E2E fixture 未整備、SOC + APBC と
  SOC + SubShape を合成した phase 経路は独立検証されていない)。
  2 組の subset は fixture を保持している:
  ``case_soc_rashba_2d_nosub_apbc`` (SOC + APBC、0.03% 差)、
  ``case_soc_rashba_2d_sub`` (SOC + SubShape、0.22% 差)。
- 2-body Sz 非保存相互作用 (spin-flip Coulomb, Hund coupling,
  pair hopping)。CoulombIntra (on-site U) のみが v3.5 の唯一の 2-body 項。

クラス一致性チェック (v3 General)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

General path では ``aggregate_general_orbital_params`` が、
平均化する **前** に ``orbitalidx_general.def`` の各クラスに割り当てられた
sign 付き F 成分が ``class_consistency_tol`` (default 1e-8) 以内で
一致していることを検査する。不一致であれば
``ClassInconsistencyError`` を offending idx と観測された max residual
とともに raise する。これにより、StdFace が生成したクラスが仮定する
対称性を Slater state が守っていないケース (対称なハミルトニアンで
自発的対称性破れの UHF 基底状態が出るなど) を silent averaging から
守る。

密度行列チェック (推奨)
^^^^^^^^^^^^^^^^^^^^^^^

``--check-density`` は (k, -k) pair 構築から 1 体密度行列を構築し、
H-wave の物理基底 ``greenone.dat`` と element-wise 比較する (許容 1e-10)。
不一致は fatal で、H-wave APBC、ブリッジ、もしくは geometry の前提に
バグがあることを示す。

SOC + SubShape > [1, 1, 1] では密度チェックは
``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
(許容 1e-10) に切り替わる (H-wave ``greenone.dat`` の fold path は
本組合せで known-buggy、``green_sublattice`` が source of truth)。
チェックは H-wave の ``green_sublattice`` を ``gauge_lift`` で物理
基底に持ち上げ、SHIPPING ``conj(A) @ A.T`` (mVMC に emit する A
そのもの) と element-wise 照合するため、shipping-A 経路の regression
は silent には通らない。これにより v3.4 で塞いだ密度 gate (dual-A
hole) が v3.5 spec §2-3 で導出した gauge lift により復活し、
v3.4 の ``_soc_reference_convention`` escape hatch は削除された。

終了コード
^^^^^^^^^^

- ``0`` — 成功
- ``2`` — fail-fast ガードが入力を拒絶 (スコープ外モード、Sz-free SCF、
  有限温度の分数占有残留、``orbitalidx.def`` / geometry / 境界条件の
  不整合など)。stderr に失敗した検査名を明示する。
- ``3`` — ``--check-density`` が許容差を超える不一致を検出
