"use strict";

const STAGES = [
  {
    title: "Виртуальный объем",
    subtitle: "Конечные целые запасы частиц внутри заданного объема."
  },
  {
    title: "Периодическая развертка",
    subtitle: "Расчет максимальной емкости выбранных атомов без перебора всех миров."
  },
  {
    title: "Химические соединения",
    subtitle: "Проверка рецептов соединений на поле ограничений атомного состава."
  },
  {
    title: "Био- и техно-классы",
    subtitle: "Вывод функциональных классов из уже рассчитанных соединений."
  },
  {
    title: "Дельта-классификация",
    subtitle: "Каждый объект проходит единственный путь по бинарному дереву."
  }
];

const PRESETS = {
  water: { volumeCells: 2048, protons: 40, neutrons: 32, electrons: 40 },
  organic: { volumeCells: 4096, protons: 520, neutrons: 560, electrons: 520 },
  tech: { volumeCells: 8192, protons: 1200, neutrons: 1320, electrons: 1200 }
};

const RULE_PACKS = {
  reference_science: {
    label: "Опорная наука",
    trustedCore: true,
    mode: "trusted_prototype"
  },
  katyushchik_hypothesis: {
    label: "Гипотеза Катющика",
    trustedCore: false,
    mode: "research_hypothesis"
  }
};

const ELEMENTS = [
  { symbol: "H", name: "Водород", protons: 1, neutrons: 0, electrons: 1, phase: "gas", color: "#76a9fa" },
  { symbol: "C", name: "Углерод", protons: 6, neutrons: 6, electrons: 6, phase: "solid", color: "#596066" },
  { symbol: "N", name: "Азот", protons: 7, neutrons: 7, electrons: 7, phase: "gas", color: "#7b6fd6" },
  { symbol: "O", name: "Кислород", protons: 8, neutrons: 8, electrons: 8, phase: "gas", color: "#d95c57" },
  { symbol: "Na", name: "Натрий", protons: 11, neutrons: 12, electrons: 11, phase: "solid", color: "#c46a2b" },
  { symbol: "Si", name: "Кремний", protons: 14, neutrons: 14, electrons: 14, phase: "solid", color: "#1f7a68" },
  { symbol: "Cl", name: "Хлор", protons: 17, neutrons: 18, electrons: 17, phase: "gas", color: "#74a857" },
  { symbol: "Fe", name: "Железо", protons: 26, neutrons: 30, electrons: 26, phase: "solid", color: "#8a4f43" }
];

const COMPOUNDS = [
  { formula: "H2", name: "Молекулярный водород", atoms: { H: 2 }, phase: "gas", family: "simple" },
  { formula: "O2", name: "Молекулярный кислород", atoms: { O: 2 }, phase: "gas", family: "simple" },
  { formula: "N2", name: "Молекулярный азот", atoms: { N: 2 }, phase: "gas", family: "simple" },
  { formula: "H2O", name: "Вода", atoms: { H: 2, O: 1 }, phase: "liquid", family: "bio" },
  { formula: "CO2", name: "Диоксид углерода", atoms: { C: 1, O: 2 }, phase: "gas", family: "bio" },
  { formula: "CH4", name: "Метан", atoms: { C: 1, H: 4 }, phase: "gas", family: "bio" },
  { formula: "NH3", name: "Аммиак", atoms: { N: 1, H: 3 }, phase: "gas", family: "bio" },
  { formula: "NaCl", name: "Хлорид натрия", atoms: { Na: 1, Cl: 1 }, phase: "solid", family: "tech" },
  { formula: "SiO2", name: "Диоксид кремния", atoms: { Si: 1, O: 2 }, phase: "solid", family: "tech" },
  { formula: "C6H12O6", name: "Глюкоза", atoms: { C: 6, H: 12, O: 6 }, phase: "solid", family: "bio" },
  { formula: "C2H5NO2", name: "Глицин", atoms: { C: 2, H: 5, N: 1, O: 2 }, phase: "solid", family: "bio" }
];

const FUNCTIONAL_CLASSES = [
  {
    id: "bio.aqueous_medium",
    label: "Водная биосреда",
    type: "bio",
    requires: { compounds: { H2O: 1 } }
  },
  {
    id: "bio.carbon_nitrogen_set",
    label: "C-H-O-N предбиохимия",
    type: "bio",
    requires: { compounds: { H2O: 1, CO2: 1, NH3: 1, CH4: 1 } }
  },
  {
    id: "bio.simple_biomolecule",
    label: "Простая биомолекула",
    type: "bio",
    requires: { compounds: { C2H5NO2: 1 } }
  },
  {
    id: "tech.electrolyte",
    label: "Электролитический контур",
    type: "tech",
    requires: { compounds: { H2O: 1, NaCl: 1 } }
  },
  {
    id: "tech.silicate",
    label: "Силикатная основа",
    type: "tech",
    requires: { compounds: { SiO2: 1 } }
  },
  {
    id: "tech.metal_stock",
    label: "Металлический запас",
    type: "tech",
    requires: { elements: { Fe: 1 } }
  }
];

const ASSEMBLY_TARGETS = [
  {
    id: "bio.minimal_context",
    label: "Минимальная биологическая среда",
    type: "bio-assembly",
    requires: {
      functions: {
        "bio.aqueous_medium": 1,
        "bio.carbon_nitrogen_set": 1,
        "bio.simple_biomolecule": 1
      }
    }
  },
  {
    id: "tech.basic_device",
    label: "Базовый технический модуль",
    type: "tech-assembly",
    requires: {
      functions: {
        "tech.electrolyte": 1,
        "tech.silicate": 1,
        "tech.metal_stock": 1
      }
    }
  },
  {
    id: "tech.logic_carrier",
    label: "Носитель логики",
    type: "tech-assembly",
    requires: {
      functions: {
        "tech.silicate": 1,
        "tech.metal_stock": 1
      }
    }
  }
];

const CONCEPT_ALIASES = {
  elements: {
    H: ["водород", "водорода", "гидроген", "hydrogen"],
    C: ["углерод", "углерода", "carbon"],
    N: ["азот", "азота", "nitrogen"],
    O: ["кислород", "кислорода", "oxygen"],
    Na: ["натрий", "натрия", "sodium"],
    Si: ["кремний", "кремния", "silicon"],
    Cl: ["хлор", "хлора", "chlorine"],
    Fe: ["железо", "железа", "iron", "металл"]
  },
  compounds: {
    H2: ["молекулярный водород"],
    O2: ["молекулярный кислород"],
    N2: ["молекулярный азот"],
    H2O: ["вода", "воды", "воду", "water", "жидкая вода"],
    CO2: ["диоксид углерода", "углекислый газ", "углекислота", "carbon dioxide"],
    CH4: ["метан", "метана", "methane"],
    NH3: ["аммиак", "аммиака", "ammonia"],
    NaCl: ["хлорид натрия", "соль", "поваренная соль", "sodium chloride"],
    SiO2: ["диоксид кремния", "кремнезем", "силикатная база", "silicon dioxide"],
    C6H12O6: ["глюкоза", "глюкозу", "glucose", "сахар"],
    C2H5NO2: ["глицин", "глицина", "glycine", "аминокислота"]
  },
  functions: {
    "bio.aqueous_medium": ["водная биосреда", "биосреда", "водная среда"],
    "bio.carbon_nitrogen_set": ["предбиохимия", "c-h-o-n", "органический набор"],
    "bio.simple_biomolecule": ["простая биомолекула", "биомолекула"],
    "tech.electrolyte": ["электролитический контур", "электролит", "ионный контур"],
    "tech.silicate": ["силикатная основа", "силикат", "изоляция", "диэлектрик"],
    "tech.metal_stock": ["металлический запас", "проводник", "металл"]
  },
  assemblies: {
    "bio.minimal_context": ["биологический объект", "биология", "минимальная биологическая среда", "живая среда"],
    "tech.basic_device": ["техническое устройство", "устройство", "базовый технический модуль", "техустройство"],
    "tech.logic_carrier": ["носитель логики", "логический носитель", "информационное тело"]
  },
  leaves: {
    Field: ["поле"],
    FreeParticles: ["свободные частицы", "частицы"],
    Subparticles: ["субчастицы"],
    Solid: ["твердое вещество", "твердое"],
    Liquid: ["жидкость"],
    Plasma: ["плазма"],
    NeutralGas: ["нейтральный газ", "неионизированный газ"]
  },
  branches: {
    Particles: ["частицы"],
    GroupsOfParticles: ["группы частиц", "сгруппированные частицы"],
    Substance: ["вещество"],
    GasOrLiquid: ["жидкости и газы", "текучие вещества"],
    Gas: ["газ", "газы"]
  }
};

const TREE = {
  Matter: {
    deltaId: "matter.continuous",
    predicate: "Continuous(x)",
    field: "continuous",
    trueNode: "Field",
    falseNode: "Particles"
  },
  Particles: {
    deltaId: "particles.grouped",
    predicate: "Grouped(x)",
    field: "grouped",
    trueNode: "GroupsOfParticles",
    falseNode: "FreeParticles"
  },
  GroupsOfParticles: {
    deltaId: "groups.planetary",
    predicate: "Planetary(x)",
    field: "planetary",
    trueNode: "Subparticles",
    falseNode: "Substance"
  },
  Substance: {
    deltaId: "substance.preserves_shape",
    predicate: "PreservesShape(x)",
    field: "preservesShape",
    trueNode: "Solid",
    falseNode: "GasOrLiquid"
  },
  GasOrLiquid: {
    deltaId: "fluid.preserves_volume",
    predicate: "PreservesVolume(x)",
    field: "preservesVolume",
    trueNode: "Liquid",
    falseNode: "Gas"
  },
  Gas: {
    deltaId: "gas.ionized",
    predicate: "Ionized(x)",
    field: "ionized",
    trueNode: "Plasma",
    falseNode: "NeutralGas"
  }
};

const LEAF_LABELS = {
  Field: "Поле",
  FreeParticles: "Свободные частицы",
  Subparticles: "Субчастицы",
  Solid: "Твердое вещество",
  Liquid: "Жидкость",
  Plasma: "Плазма",
  NeutralGas: "Нейтральный газ"
};

const PHASE_LABELS = {
  solid: "твердое",
  liquid: "жидкость",
  gas: "газ"
};

let manifest = null;
let exactRun = null;
let conceptCatalog = null;
let computedSpace = null;
let stageIndex = 0;
let currentState = null;
let timerId = null;
let eventItems = [];
let activeQueryText = "";
let activeQueryResult = null;

const els = {
  volumeCells: document.getElementById("volumeCells"),
  protons: document.getElementById("protons"),
  neutrons: document.getElementById("neutrons"),
  electrons: document.getElementById("electrons"),
  ionizedGases: document.getElementById("ionizedGases"),
  rulePack: document.getElementById("rulePack"),
  userQuery: document.getElementById("userQuery"),
  queryBtn: document.getElementById("queryBtn"),
  queryResult: document.getElementById("queryResult"),
  runBtn: document.getElementById("runBtn"),
  stepBtn: document.getElementById("stepBtn"),
  resetBtn: document.getElementById("resetBtn"),
  stageList: document.getElementById("stageList"),
  stageTitle: document.getElementById("stageTitle"),
  stageSubtitle: document.getElementById("stageSubtitle"),
  statusChip: document.getElementById("statusChip"),
  canvas: document.getElementById("volumeCanvas"),
  summary: document.getElementById("summary"),
  eventLog: document.getElementById("eventLog"),
  mermaidText: document.getElementById("mermaidText"),
  manifestText: document.getElementById("manifestText"),
  exactText: document.getElementById("exactText"),
  exactSummary: document.getElementById("exactSummary"),
  computedText: document.getElementById("computedText"),
  computedSummary: document.getElementById("computedSummary"),
  treeLegend: document.getElementById("treeLegend"),
  classificationTable: document.getElementById("classificationTable")
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  manifest = await loadManifest();
  exactRun = await loadExactRun();
  conceptCatalog = await loadConceptCatalog();
  computedSpace = await loadComputedSpace();
  mergeConfirmedConcepts(conceptCatalog);
  currentState = computeState();
  bindEvents();
  renderManifest();
  renderExactRun();
  renderComputedSpace();
  render();
}

async function loadManifest() {
  try {
    const response = await fetch("data/matter-deltas.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    return {
      mermaid: "graph TD\n  M[Материя] -->|Continuous(x)| F[Поле]\n  M -->|not Continuous(x)| P[Частицы]",
      deltas: []
    };
  }
}

async function loadExactRun() {
  try {
    const response = await fetch("data/exact-latest.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    return null;
  }
}

async function loadConceptCatalog() {
  try {
    const response = await fetch("data/confirmed-concepts.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    return null;
  }
}

async function loadComputedSpace() {
  try {
    const response = await fetch("data/computed-compound-space.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    return null;
  }
}

function mergeConfirmedConcepts(catalog) {
  if (!catalog) {
    return;
  }
  const existingElements = new Set(ELEMENTS.map((element) => element.symbol));
  (catalog.elements || []).forEach((element) => {
    const aliases = uniqueStrings(element.aliases || []);
    CONCEPT_ALIASES.elements[element.symbol] = uniqueStrings([
      ...(CONCEPT_ALIASES.elements[element.symbol] || []),
      ...aliases
    ]);
    if (existingElements.has(element.symbol)) {
      const existing = ELEMENTS.find((item) => item.symbol === element.symbol);
      if (existing) {
        existing.source = "pubchem.periodic_table";
        existing.atomicMass = element.atomic_mass;
        existing.groupBlock = element.group_block;
      }
      return;
    }
    ELEMENTS.push({
      symbol: element.symbol,
      name: element.name_en,
      protons: element.atomic_number,
      neutrons: element.model.neutrons,
      electrons: element.atomic_number,
      phase: element.model.phase,
      color: colorForElement(element.symbol),
      source: "pubchem.periodic_table",
      atomicMass: element.atomic_mass,
      groupBlock: element.group_block,
      modelStatus: element.model.status
    });
    existingElements.add(element.symbol);
  });

  (catalog.compounds || []).forEach((compound) => {
    const aliases = uniqueStrings(compound.aliases || []);
    const previousAliases = CONCEPT_ALIASES.compounds[compound.formula] || [];
    const atoms = parseFormula(compound.formula);
    if (!atoms) {
      return;
    }

    const existing = COMPOUNDS.find((item) => {
      if (item.cid && item.cid === compound.cid) {
        return true;
      }
      const localAliases = item.aliases || previousAliases;
      return item.formula === compound.formula
        && (normalizeQuery(item.name) === normalizeQuery(compound.query)
          || localAliases.some((alias) => normalizeQuery(alias) === normalizeQuery(compound.query)));
    });

    if (existing) {
      existing.source = "pubchem.compound_property";
      existing.cid = compound.cid;
      existing.conceptId = `cid:${compound.cid}`;
      existing.iupacName = compound.iupac_name;
      existing.pubchemFormula = compound.pubchem_molecular_formula;
      existing.aliases = uniqueStrings([...(existing.aliases || []), ...aliases]);
      CONCEPT_ALIASES.compounds[compound.formula] = uniqueStrings([...previousAliases, ...aliases]);
      return;
    }

    COMPOUNDS.push({
      conceptId: `cid:${compound.cid}`,
      formula: compound.formula,
      name: compound.query,
      atoms,
      aliases,
      phase: compound.model.phase,
      family: compound.model.family,
      source: "pubchem.compound_property",
      cid: compound.cid,
      iupacName: compound.iupac_name,
      pubchemFormula: compound.pubchem_molecular_formula,
      modelStatus: compound.model.status
    });
    CONCEPT_ALIASES.compounds[compound.formula] = uniqueStrings([...previousAliases, ...aliases]);
  });
}

function parseFormula(formula) {
  const stack = [{}];
  for (let index = 0; index < formula.length;) {
    const char = formula[index];
    if (char === "(") {
      stack.push({});
      index += 1;
      continue;
    }
    if (char === ")") {
      if (stack.length === 1) {
        return null;
      }
      index += 1;
      const countMatch = formula.slice(index).match(/^(\d*)/);
      const multiplier = countMatch && countMatch[1] ? Number.parseInt(countMatch[1], 10) : 1;
      index += countMatch ? countMatch[0].length : 0;
      const group = stack.pop();
      const target = stack[stack.length - 1];
      Object.entries(group).forEach(([symbol, count]) => {
        target[symbol] = (target[symbol] || 0) + count * multiplier;
      });
      continue;
    }
    const match = formula.slice(index).match(/^([A-Z][a-z]?)(\d*)/);
    if (!match) {
      return null;
    }
    const symbol = match[1];
    const count = match[2] ? Number.parseInt(match[2], 10) : 1;
    const target = stack[stack.length - 1];
    target[symbol] = (target[symbol] || 0) + count;
    index += match[0].length;
  }
  if (stack.length !== 1) {
    return null;
  }
  return stack[0];
}

function uniqueStrings(values) {
  return [...new Set(values.filter((value) => typeof value === "string" && value.trim()).map((value) => value.trim()))];
}

function colorForElement(symbol) {
  const palette = ["#1f7a68", "#c46a2b", "#445d9c", "#9a5b10", "#596066", "#237a3b", "#7b6fd6", "#d95c57"];
  let hash = 0;
  for (let i = 0; i < symbol.length; i += 1) {
    hash = (hash * 31 + symbol.charCodeAt(i)) >>> 0;
  }
  return palette[hash % palette.length];
}

function bindEvents() {
  els.runBtn.addEventListener("click", runProcess);
  els.stepBtn.addEventListener("click", stepProcess);
  els.resetBtn.addEventListener("click", resetProcess);
  els.queryBtn.addEventListener("click", runUserQuery);
  els.userQuery.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      runUserQuery();
    }
  });

  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
  });

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });

  [els.volumeCells, els.protons, els.neutrons, els.electrons, els.ionizedGases, els.rulePack].forEach((input) => {
    input.addEventListener("input", () => {
      currentState = computeState();
      refreshActiveQuery();
      render();
    });
  });

  window.addEventListener("resize", () => renderCanvas());
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) {
    return;
  }
  els.volumeCells.value = preset.volumeCells;
  els.protons.value = preset.protons;
  els.neutrons.value = preset.neutrons;
  els.electrons.value = preset.electrons;
  resetProcess();
}

function setTab(name) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll(".tab-view").forEach((view) => {
    view.classList.toggle("active", view.id === `${name}View`);
  });
}

function runProcess() {
  if (timerId) {
    stopTimer();
    return;
  }
  if (stageIndex >= STAGES.length - 1) {
    stageIndex = 0;
    eventItems = [];
  }
  els.runBtn.textContent = "Пауза";
  els.statusChip.textContent = "расчет";
  els.statusChip.className = "status-chip running";
  logCurrentStage();
  timerId = window.setInterval(() => {
    if (stageIndex >= STAGES.length - 1) {
      stopTimer(true);
      return;
    }
    stageIndex += 1;
    currentState = computeState();
    refreshActiveQuery();
    logCurrentStage();
    render();
  }, 850);
}

function stepProcess() {
  stopTimer();
  if (stageIndex < STAGES.length - 1) {
    stageIndex += 1;
  }
  currentState = computeState();
  refreshActiveQuery();
  logCurrentStage();
  render();
}

function resetProcess() {
  stopTimer();
  stageIndex = 0;
  eventItems = [];
  currentState = computeState();
  refreshActiveQuery();
  render();
}

function runUserQuery() {
  activeQueryText = els.userQuery.value.trim();
  activeQueryResult = answerUserQuery(activeQueryText, currentState || computeState());
  if (activeQueryText) {
    eventItems.push({
      stage: "Запрос",
      message: activeQueryResult.status === "ok"
        ? `Найдено понятие: ${activeQueryResult.target.label}; емкость ${formatInt(activeQueryResult.capacity)}.`
        : activeQueryResult.message
    });
    if (eventItems.length > 12) {
      eventItems = eventItems.slice(-12);
    }
  }
  render();
}

function refreshActiveQuery() {
  if (!activeQueryText) {
    activeQueryResult = null;
    return;
  }
  activeQueryResult = answerUserQuery(activeQueryText, currentState || computeState());
}

function stopTimer(done = false) {
  if (timerId) {
    window.clearInterval(timerId);
    timerId = null;
  }
  els.runBtn.textContent = "Запустить";
  els.statusChip.textContent = done ? "завершено" : "ожидание";
  els.statusChip.className = done ? "status-chip done" : "status-chip";
}

function readInput() {
  const selectedRulePack = RULE_PACKS[els.rulePack.value] ? els.rulePack.value : "reference_science";
  return {
    volumeCells: sanitizeInteger(els.volumeCells.value, 1, 1000000),
    protons: sanitizeInteger(els.protons.value, 0, 1000000000),
    neutrons: sanitizeInteger(els.neutrons.value, 0, 1000000000),
    electrons: sanitizeInteger(els.electrons.value, 0, 1000000000),
    ionizedGases: els.ionizedGases.checked,
    rulePack: selectedRulePack
  };
}

function sanitizeInteger(value, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.max(min, Math.min(max, parsed));
}

function computeState() {
  const input = readInput();
  const rulePack = RULE_PACKS[input.rulePack];
  const elementRows = ELEMENTS.map((element) => {
    const maxAtoms = maxAtomsForElement(input, element);
    return { ...element, maxAtoms };
  });
  const elementMap = Object.fromEntries(elementRows.map((row) => [row.symbol, row.maxAtoms]));
  const compoundRows = COMPOUNDS.map((compound) => {
    const maxMolecules = maxUnitsForAtomRecipe(compound.atoms, input);
    return { ...compound, maxMolecules };
  });
  const compoundMap = Object.fromEntries(compoundRows.map((row) => [row.formula, row.maxMolecules]));
  const functionRows = FUNCTIONAL_CLASSES.map((item) => {
    const capacity = maxUnitsForRequirements(item.requires, input);
    return { ...item, capacity, possible: capacity > 0 };
  });
  const functionMap = Object.fromEntries(functionRows.map((row) => [row.id, row.capacity]));
  const assemblyRows = ASSEMBLY_TARGETS.map((item) => {
    const capacity = maxUnitsForRequirements(item.requires, input);
    return { ...item, capacity, possible: capacity > 0 };
  });
  const matterObjects = buildMatterObjects(input, elementRows, compoundRows);
  const classifications = matterObjects.map((object) => ({
    ...object,
    classification: classifyMatterObject(object)
  }));
  const canonical = canonicalState(input, elementRows, compoundRows, functionRows, assemblyRows, classifications);
  const stateHash = hashCanonical(canonical);

  return {
    input,
    rulePack,
    elements: elementRows,
    elementMap,
    compounds: compoundRows,
    compoundMap,
    functions: functionRows,
    functionMap,
    assemblies: assemblyRows,
    classifications,
    canonical,
    stateHash
  };
}

function maxAtomsForElement(input, element) {
  const protonLimit = Math.floor(input.protons / element.protons);
  const neutronLimit = element.neutrons === 0 ? Number.POSITIVE_INFINITY : Math.floor(input.neutrons / element.neutrons);
  const electronLimit = Math.floor(input.electrons / element.electrons);
  return finiteMinimum([protonLimit, neutronLimit, electronLimit]);
}

function finiteMinimum(values) {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (!finiteValues.length) {
    return 0;
  }
  return Math.max(0, Math.min(...finiteValues));
}

function maxUnitsForAtomRecipe(recipe, input) {
  return maxUnitsForParticleRequirement(particleRequirementsForAtomRecipe(recipe), input);
}

function maxUnitsForRequirements(requirements, input) {
  return maxUnitsForParticleRequirement(particleRequirementsForRequirements(requirements), input);
}

function maxUnitsForParticleRequirement(requirement, input) {
  const limits = [];
  if (requirement.protons > 0) {
    limits.push(Math.floor(input.protons / requirement.protons));
  }
  if (requirement.neutrons > 0) {
    limits.push(Math.floor(input.neutrons / requirement.neutrons));
  }
  if (requirement.electrons > 0) {
    limits.push(Math.floor(input.electrons / requirement.electrons));
  }
  return finiteMinimum(limits);
}

function particleRequirementsForRequirements(requirements) {
  const total = emptyRequirement();
  Object.entries(requirements.elements || {}).forEach(([symbol, count]) => {
    addRequirement(total, particleRequirementsForElement(symbol, count));
  });
  Object.entries(requirements.compounds || {}).forEach(([formula, count]) => {
    addRequirement(total, particleRequirementsForCompound(formula, count));
  });
  Object.entries(requirements.functions || {}).forEach(([id, count]) => {
    const item = FUNCTIONAL_CLASSES.find((row) => row.id === id);
    if (item) {
      addRequirement(total, scaleRequirement(particleRequirementsForRequirements(item.requires), count));
    }
  });
  return total;
}

function particleRequirementsForAtomRecipe(recipe) {
  const total = emptyRequirement();
  Object.entries(recipe).forEach(([symbol, count]) => {
    addRequirement(total, particleRequirementsForElement(symbol, count));
  });
  return total;
}

function particleRequirementsForCompound(formula, count = 1) {
  const compound = COMPOUNDS.find((row) => row.formula === formula);
  return compound ? scaleRequirement(particleRequirementsForAtomRecipe(compound.atoms), count) : emptyRequirement();
}

function particleRequirementsForElement(symbol, count = 1) {
  const element = ELEMENTS.find((row) => row.symbol === symbol);
  if (!element) {
    return emptyRequirement();
  }
  return {
    protons: element.protons * count,
    neutrons: element.neutrons * count,
    electrons: element.electrons * count
  };
}

function emptyRequirement() {
  return { protons: 0, neutrons: 0, electrons: 0 };
}

function addRequirement(target, addition) {
  target.protons += addition.protons;
  target.neutrons += addition.neutrons;
  target.electrons += addition.electrons;
  return target;
}

function scaleRequirement(requirement, count) {
  return {
    protons: requirement.protons * count,
    neutrons: requirement.neutrons * count,
    electrons: requirement.electrons * count
  };
}

function canonicalState(input, elements, compounds, functions, assemblies, classifications) {
  return {
    schema: "logical-delta-matter-state@0.1",
    rulePack: input.rulePack,
    input: {
      volumeCells: input.volumeCells,
      protons: input.protons,
      neutrons: input.neutrons,
      electrons: input.electrons,
      ionizedGases: input.ionizedGases
    },
    elements: elements
      .filter((row) => row.maxAtoms > 0)
      .map((row) => [row.symbol, row.maxAtoms]),
    compounds: compounds
      .filter((row) => row.maxMolecules > 0)
      .map((row) => [row.formula, row.maxMolecules]),
    functions: functions
      .filter((row) => row.possible)
      .map((row) => [row.id, row.capacity]),
    assemblies: assemblies
      .filter((row) => row.possible)
      .map((row) => [row.id, row.capacity]),
    leaves: Object.entries(countBy(classifications, (row) => row.classification.leaf)).sort(([a], [b]) => a.localeCompare(b))
  };
}

function hashCanonical(value) {
  const text = stableStringify(value);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function stableStringify(value) {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function buildMatterObjects(input, elements, compounds) {
  const objects = [
    {
      id: "field.background",
      label: "Фон поля",
      kind: "field",
      count: 1,
      traits: {
        continuous: true,
        grouped: false,
        planetary: false,
        preservesShape: false,
        preservesVolume: false,
        ionized: false
      }
    }
  ];

  if (input.protons > 0) {
    objects.push(freeParticleObject("particle.proton", "Протон", input.protons));
  }
  if (input.neutrons > 0) {
    objects.push(freeParticleObject("particle.neutron", "Нейтрон", input.neutrons));
  }
  if (input.electrons > 0) {
    objects.push(freeParticleObject("particle.electron", "Электрон", input.electrons));
  }

  elements.filter((row) => row.maxAtoms > 0).forEach((row) => {
    objects.push({
      id: `element.${row.symbol}`,
      label: `${row.name} (${row.symbol})`,
      kind: "element",
      count: row.maxAtoms,
      traits: traitsForPhase(row.phase, input.ionizedGases)
    });
  });

  compounds.filter((row) => row.maxMolecules > 0).forEach((row) => {
    objects.push({
      id: `compound.${row.formula}`,
      label: `${row.name} (${row.formula})`,
      kind: "compound",
      count: row.maxMolecules,
      traits: traitsForPhase(row.phase, input.ionizedGases)
    });
  });

  return objects;
}

function freeParticleObject(id, label, count) {
  return {
    id,
    label,
    kind: "free-particle",
    count,
    traits: {
      continuous: false,
      grouped: false,
      planetary: false,
      preservesShape: false,
      preservesVolume: false,
      ionized: false
    }
  };
}

function traitsForPhase(phase, ionizedGases) {
  return {
    continuous: false,
    grouped: true,
    planetary: false,
    preservesShape: phase === "solid",
    preservesVolume: phase === "solid" || phase === "liquid",
    ionized: phase === "gas" && ionizedGases
  };
}

function classifyMatterObject(object) {
  let node = "Matter";
  const path = [];
  while (TREE[node]) {
    const delta = TREE[node];
    const value = readBooleanTrait(object, delta.field);
    const nextNode = value ? delta.trueNode : delta.falseNode;
    path.push({
      delta_id: delta.deltaId,
      predicate: delta.predicate,
      value,
      branch: nextNode
    });
    node = nextNode;
  }
  return {
    leaf: node,
    label: LEAF_LABELS[node] || node,
    path
  };
}

function readBooleanTrait(object, field) {
  const value = object.traits[field];
  if (typeof value !== "boolean") {
    throw new Error(`Object ${object.id} violates totality for ${field}`);
  }
  return value;
}

function answerUserQuery(text, state) {
  if (!text.trim()) {
    return {
      status: "empty",
      message: "Введите целевой объект: например, вода, глюкоза, проводник или техническое устройство."
    };
  }
  const matches = resolveConcept(text);
  if (!matches.length) {
    return {
      status: "unknown",
      message: "UNKNOWN: в текущем словаре нет однозначного понятия для этого запроса.",
      suggestions: conceptSuggestions()
    };
  }
  const topScore = matches[0].score;
  const winners = matches.filter((match) => match.score === topScore);
  if (winners.length > 1) {
    return {
      status: "ambiguous",
      message: "UNKNOWN: запрос неоднозначен в текущем словаре.",
      candidates: winners.map((match) => match.entry.label)
    };
  }
  return inspectConcept(winners[0].entry, state);
}

function resolveConcept(text) {
  const normalized = normalizeQuery(text);
  return buildConceptRegistry()
    .map((entry) => {
      const best = entry.aliases
        .map((alias) => ({ alias, normalized: normalizeQuery(alias) }))
        .filter((alias) => alias.normalized)
        .map((alias) => ({
          alias: alias.alias,
          score: aliasScore(normalized, alias.normalized)
        }))
        .filter((alias) => alias.score > 0)
        .sort((a, b) => b.score - a.score)[0];
      return best ? { entry, score: best.score, alias: best.alias } : null;
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score || b.alias.length - a.alias.length);
}

function buildConceptRegistry() {
  const entries = [];
  ELEMENTS.forEach((element) => {
    entries.push({
      kind: "element",
      id: element.symbol,
      label: `${element.name} (${element.symbol})`,
      aliases: [element.symbol, element.name, ...(CONCEPT_ALIASES.elements[element.symbol] || [])],
      source: element.source || "local.prototype.elements"
    });
  });
  COMPOUNDS.forEach((compound) => {
    const conceptId = compound.conceptId || compound.formula;
    entries.push({
      kind: "compound",
      id: conceptId,
      formula: compound.formula,
      label: `${compound.name} (${compound.formula})`,
      aliases: [compound.formula, compound.name, ...(compound.aliases || CONCEPT_ALIASES.compounds[compound.formula] || [])],
      source: compound.source || "local.prototype.compounds"
    });
  });
  const knownCompoundFormulas = new Set(COMPOUNDS.map((compound) => compound.formula));
  computedCandidatesForRegistry().forEach((candidate) => {
    if (knownCompoundFormulas.has(candidate.formula)) {
      return;
    }
    entries.push({
      kind: "computed_compound",
      id: candidate.candidate_id,
      formula: candidate.formula,
      atoms: candidate.atoms,
      label: `${candidate.formula} (${candidate.status})`,
      aliases: [candidate.formula, candidate.candidate_id],
      source: "katyushchik_hypothesis.compound_space",
      computedCandidate: candidate
    });
  });
  FUNCTIONAL_CLASSES.forEach((item) => {
    entries.push({
      kind: "function",
      id: item.id,
      label: item.label,
      aliases: [item.label, item.id, ...(CONCEPT_ALIASES.functions[item.id] || [])],
      source: "local.prototype.functions"
    });
  });
  ASSEMBLY_TARGETS.forEach((item) => {
    entries.push({
      kind: "assembly",
      id: item.id,
      label: item.label,
      aliases: [item.label, item.id, ...(CONCEPT_ALIASES.assemblies[item.id] || [])],
      source: "local.prototype.assemblies"
    });
  });
  Object.entries(CONCEPT_ALIASES.leaves).forEach(([id, aliases]) => {
    entries.push({
      kind: "leaf",
      id,
      label: LEAF_LABELS[id] || id,
      aliases: [LEAF_LABELS[id] || id, id, ...aliases]
    });
  });
  Object.entries(CONCEPT_ALIASES.branches).forEach(([id, aliases]) => {
    entries.push({
      kind: "branch",
      id,
      label: branchLabel(id),
      aliases: [branchLabel(id), id, ...aliases]
    });
  });
  return entries;
}

function computedCandidatesForRegistry() {
  if (!computedSpace || !Array.isArray(computedSpace.candidates)) {
    return [];
  }
  const seen = new Set();
  const rows = [];
  computedSpace.candidates.forEach((candidate) => {
    if (!candidate.formula || !candidate.atoms || seen.has(candidate.formula)) {
      return;
    }
    seen.add(candidate.formula);
    rows.push(candidate);
  });
  return rows;
}

function theoryVerdictForFormula(formula, state) {
  if (!state || state.input.rulePack !== "katyushchik_hypothesis") {
    return null;
  }
  return computedCandidateForFormula(formula);
}

function computedCandidateForFormula(formula) {
  if (!computedSpace || !Array.isArray(computedSpace.candidates)) {
    return null;
  }
  const matches = computedSpace.candidates.filter((candidate) => candidate.formula === formula);
  return matches.find((candidate) => candidate.confirmed)
    || matches.find((candidate) => candidate.origin === "katyushchik_generated")
    || matches[0]
    || null;
}

function theoryHasStructuralBlock(candidate) {
  if (!candidate || candidate.status !== "impossible") {
    return false;
  }
  return (candidate.reasons || []).some((reason) => reason !== "resource_insufficient");
}

function inspectConcept(target, state) {
  if (target.kind === "element") {
    const element = ELEMENTS.find((row) => row.symbol === target.id);
    const requirement = particleRequirementsForElement(target.id, 1);
    const capacity = state.elementMap[target.id] || 0;
    return queryOk(target, capacity, requirement, state, {
      definition: `Элемент ${element.name} задается атомным номером Z=${element.protons}.`,
      chain: ["ParticleStock", `Element(${target.id})`],
      classification: classifyTargetPhase(element.phase, state.input.ionizedGases)
    });
  }
  if (target.kind === "compound") {
    const formula = target.formula || target.id;
    const compound = COMPOUNDS.find((row) => (row.conceptId || row.formula) === target.id) || COMPOUNDS.find((row) => row.formula === formula);
    const requirement = particleRequirementsForCompound(formula, 1);
    const theoryVerdict = theoryVerdictForFormula(formula, state);
    const structuralBlocked = theoryHasStructuralBlock(theoryVerdict);
    const capacity = structuralBlocked ? 0 : state.compoundMap[formula] || 0;
    return queryOk(target, capacity, requirement, state, {
      definition: `${compound.name}: ${formulaFromAtoms(compound.atoms)}.`,
      chain: ["ParticleStock", "ElementCapacity", `Compound(${formula})`],
      classification: classifyTargetPhase(compound.phase, state.input.ionizedGases),
      theoryVerdict,
      structuralBlocked
    });
  }
  if (target.kind === "computed_compound") {
    const atoms = target.atoms || parseFormula(target.formula) || {};
    const requirement = particleRequirementsForAtomRecipe(atoms);
    const theoryVerdict = target.computedCandidate || computedCandidateForFormula(target.formula);
    const structuralBlocked = theoryHasStructuralBlock(theoryVerdict);
    const resourceCapacity = maxUnitsForAtomRecipe(atoms, state.input);
    const capacity = structuralBlocked ? 0 : resourceCapacity;
    return queryOk(target, capacity, requirement, state, {
      definition: `Вычисленная формула ${target.formula}: ${formulaFromAtoms(atoms)}.`,
      chain: ["ParticleStock", "KatyushchikRulePack", `ComputedCompound(${target.formula})`],
      theoryVerdict,
      structuralBlocked
    });
  }
  if (target.kind === "function") {
    const item = FUNCTIONAL_CLASSES.find((row) => row.id === target.id);
    const requirement = particleRequirementsForRequirements(item.requires);
    const capacity = state.functionMap[target.id] || 0;
    return queryOk(target, capacity, requirement, state, {
      definition: `Функциональный класс: ${requirementsText(item.requires)}.`,
      chain: ["ParticleStock", "CompoundCapacity", `Function(${target.id})`]
    });
  }
  if (target.kind === "assembly") {
    const item = ASSEMBLY_TARGETS.find((row) => row.id === target.id);
    const requirement = particleRequirementsForRequirements(item.requires);
    const capacity = item ? maxUnitsForRequirements(item.requires, state.input) : 0;
    return queryOk(target, capacity, requirement, state, {
      definition: `Сборочная цель: ${requirementsText(item.requires)}.`,
      chain: ["ParticleStock", "FunctionCapacity", `Assembly(${target.id})`]
    });
  }
  if (target.kind === "leaf") {
    const matching = state.classifications.filter((row) => row.classification.leaf === target.id);
    return queryOk(target, matching.length, emptyRequirement(), state, {
      definition: `Лист классификации: ${target.label}.`,
      chain: ["MatterDeltaTree", `Leaf(${target.id})`],
      objects: matching.map((row) => row.label)
    });
  }
  const matching = state.classifications.filter((row) => classificationTouches(row.classification, target.id));
  return queryOk(target, matching.length, emptyRequirement(), state, {
    definition: `Промежуточная ветвь классификации: ${target.label}.`,
    chain: ["MatterDeltaTree", `Branch(${target.id})`],
    objects: matching.map((row) => row.label)
  });
}

function queryOk(target, capacity, requirement, state, details) {
  return {
    status: "ok",
    target,
    capacity,
    requirement,
    shortage: particleShortage(requirement, state.input),
    stateHash: state.stateHash,
    rulePack: state.rulePack.label,
    source: target.source || "local.prototype",
    ...details
  };
}

function classifyTargetPhase(phase, ionizedGases) {
  const object = {
    id: `query.${phase}`,
    traits: traitsForPhase(phase, ionizedGases)
  };
  return classifyMatterObject(object);
}

function classificationTouches(classification, nodeId) {
  return classification.leaf === nodeId || classification.path.some((step) => step.branch === nodeId);
}

function particleShortage(requirement, input) {
  return {
    protons: Math.max(0, requirement.protons - input.protons),
    neutrons: Math.max(0, requirement.neutrons - input.neutrons),
    electrons: Math.max(0, requirement.electrons - input.electrons)
  };
}

function formulaFromAtoms(atoms) {
  return Object.entries(atoms).map(([symbol, count]) => `${symbol}${count > 1 ? count : ""}`).join("");
}

function requirementsText(requirements) {
  const parts = [];
  Object.entries(requirements.elements || {}).forEach(([symbol, count]) => parts.push(`${symbol} x ${count}`));
  Object.entries(requirements.compounds || {}).forEach(([formula, count]) => parts.push(`${formula} x ${count}`));
  Object.entries(requirements.functions || {}).forEach(([id, count]) => {
    const item = FUNCTIONAL_CLASSES.find((row) => row.id === id);
    parts.push(`${item ? item.label : id} x ${count}`);
  });
  return parts.join(", ");
}

function branchLabel(id) {
  const labels = {
    Particles: "Частицы",
    GroupsOfParticles: "Группы частиц",
    Substance: "Вещество",
    GasOrLiquid: "Жидкости и газы",
    Gas: "Газ"
  };
  return labels[id] || id;
}

function conceptSuggestions() {
  return ["вода", "кислород", "глюкоза", "глицин", "проводник", "техническое устройство"];
}

function normalizeQuery(text) {
  return text
    .toLowerCase()
    .replaceAll("ё", "е")
    .replace(/[^a-zа-я0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function aliasScore(query, alias) {
  if (query === alias) {
    return 10000 + alias.length;
  }
  const wrappedQuery = ` ${query} `;
  const wrappedAlias = ` ${alias} `;
  if (wrappedQuery.includes(wrappedAlias)) {
    return 1000 + alias.length;
  }
  return 0;
}

function logCurrentStage() {
  const stage = STAGES[stageIndex];
  const message = stageMessage(stageIndex, currentState);
  eventItems.push({ stage: stage.title, message });
  if (eventItems.length > 12) {
    eventItems = eventItems.slice(-12);
  }
}

function stageMessage(index, state) {
  if (index === 0) {
    const total = state.input.protons + state.input.neutrons + state.input.electrons;
    return `В объеме ${state.input.volumeCells} ячеек задано ${formatInt(total)} частиц; rule-pack: ${state.rulePack.label}.`;
  }
  if (index === 1) {
    const possible = state.elements.filter((row) => row.maxAtoms > 0).length;
    return `Найдено ${possible} допустимых атомных емкостей из ${ELEMENTS.length}.`;
  }
  if (index === 2) {
    const possible = state.compounds.filter((row) => row.maxMolecules > 0).length;
    return `Проверено ${COMPOUNDS.length} рецептов, возможно ${possible} соединений.`;
  }
  if (index === 3) {
    const possible = state.functions.filter((row) => row.possible).length;
    const assemblies = state.assemblies.filter((row) => row.possible).length;
    return `Функциональных классов: ${possible}; сборочных целей: ${assemblies}.`;
  }
  const leaves = countBy(state.classifications, (row) => row.classification.label);
  return `Классификация завершена: ${Object.keys(leaves).length} активных листьев; хэш ${state.stateHash}.`;
}

function renderManifest() {
  els.mermaidText.textContent = manifest.mermaid || "";
  els.manifestText.textContent = JSON.stringify(manifest, null, 2);
  els.treeLegend.replaceChildren(
    ...((manifest.deltas || []).map((delta) => {
      const row = document.createElement("div");
      row.className = "delta-row";
      const title = document.createElement("strong");
      title.textContent = delta.delta_id;
      const body = document.createElement("span");
      const evidence = (delta.evidence_source || []).map((item) => item.source_id).join(", ");
      body.textContent = `${delta.predicate}: true -> ${delta.branches.true}, false -> ${delta.branches.false}; source: ${evidence || "local"}`;
      row.append(title, body);
      return row;
    }))
  );
}

function renderExactRun() {
  if (!els.exactText || !els.exactSummary) {
    return;
  }
  if (!exactRun) {
    const row = document.createElement("div");
    row.className = "class-row";
    const title = document.createElement("strong");
    title.textContent = "Нет данных exact-прогона";
    const body = document.createElement("span");
    body.textContent = "Запустите: python3 -m logical_delta_matter.cli --config configs/exact_mvp.json --out runs/exact_mvp --export-ui data/exact-latest.json";
    row.append(title, body);
    els.exactSummary.replaceChildren(row);
    els.exactText.textContent = "";
    return;
  }

  const metrics = exactRun.metrics || {};
  const rows = [
    ["step", metrics.step],
    ["hash", metrics.state_hash],
    ["particles", metrics.n_particles],
    ["clusters", metrics.n_clusters],
    ["promoted", metrics.n_promoted],
    ["occupancy_permille", metrics.occupancy_permille]
  ].map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "class-row";
    const title = document.createElement("strong");
    title.textContent = String(label);
    const body = document.createElement("span");
    body.textContent = String(value ?? "n/a");
    row.append(title, body);
    return row;
  });
  els.exactSummary.replaceChildren(...rows);
  els.exactText.textContent = JSON.stringify(exactRun, null, 2);
}

function renderComputedSpace() {
  if (!els.computedText || !els.computedSummary) {
    return;
  }
  if (!computedSpace) {
    const row = document.createElement("div");
    row.className = "class-row";
    const title = document.createElement("strong");
    title.textContent = "Нет вычисленного пространства соединений";
    const body = document.createElement("span");
    body.textContent = "Запустите: python3 -m logical_delta_matter.compound_cli --concepts data/confirmed-concepts.json --config data/compound-space-config.json --out data/computed-compound-space.json";
    row.append(title, body);
    els.computedSummary.replaceChildren(row);
    els.computedText.textContent = "";
    return;
  }

  const summary = computedSpace.summary || {};
  const possible = (computedSpace.candidates || []).filter((row) => row.status === "possible").slice(0, 8);
  const impossible = (computedSpace.candidates || []).filter((row) => row.status === "impossible").slice(0, 8);
  const rows = [
    ["schema", computedSpace.schema_version],
    ["rule_pack", computedSpace.rule_pack_id],
    ["total", formatInt(summary.total || 0)],
    ["possible", formatInt(summary.possible || 0)],
    ["impossible", formatInt(summary.impossible || 0)],
    ["confirmed", formatInt(summary.confirmed || 0)],
    ["generated", formatInt(summary.generated || 0)],
    ["possible_preview", possible.map((row) => row.formula).join(", ") || "нет"],
    ["impossible_preview", impossible.map((row) => `${row.formula}: ${(row.reasons || []).join("|")}`).join(", ") || "нет"]
  ].map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "class-row";
    const title = document.createElement("strong");
    title.textContent = String(label);
    const body = document.createElement("span");
    body.textContent = String(value ?? "n/a");
    row.append(title, body);
    return row;
  });

  const previewLimit = 250;
  const preview = {
    ...computedSpace,
    candidates: (computedSpace.candidates || []).slice(0, previewLimit),
    preview_note: `UI preview contains first ${previewLimit} candidates. Full JSON is stored in data/computed-compound-space.json.`
  };
  els.computedSummary.replaceChildren(...rows);
  els.computedText.textContent = JSON.stringify(preview, null, 2);
}

function render() {
  currentState = currentState || computeState();
  renderStages();
  renderHeader();
  renderSummary();
  renderQueryResult();
  renderEventLog();
  renderClassifications();
  renderCanvas();
}

function renderStages() {
  const items = STAGES.map((stage, index) => {
    const row = document.createElement("div");
    row.className = `stage-item${index === stageIndex ? " active" : ""}${index < stageIndex ? " done" : ""}`;
    const badge = document.createElement("span");
    badge.className = "stage-index";
    badge.textContent = String(index + 1);
    const label = document.createElement("span");
    label.textContent = stage.title;
    row.append(badge, label);
    return row;
  });
  els.stageList.replaceChildren(...items);
}

function renderHeader() {
  const stage = STAGES[stageIndex];
  els.stageTitle.textContent = stage.title;
  els.stageSubtitle.textContent = stage.subtitle;
}

function renderSummary() {
  const state = currentState;
  const totalParticles = state.input.protons + state.input.neutrons + state.input.electrons;
  const possibleElements = state.elements.filter((row) => row.maxAtoms > 0);
  const possibleCompounds = state.compounds.filter((row) => row.maxMolecules > 0);
  const possibleFunctions = state.functions.filter((row) => row.possible);
  const possibleAssemblies = state.assemblies.filter((row) => row.possible);
  const leafCounts = countBy(state.classifications, (row) => row.classification.label);

  const root = document.createDocumentFragment();
  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  metrics.append(
    metric("Частиц", formatInt(totalParticles)),
    metric("Элементов", String(possibleElements.length)),
    metric("Соединений", String(possibleCompounds.length)),
    metric("Rule-pack", state.rulePack.label)
  );
  root.append(metrics);

  root.append(sectionWithPills("Каноническое состояние", [state.stateHash, state.rulePack.mode], state.rulePack.trustedCore ? "good" : "warn"));
  root.append(sectionWithPills("Атомные емкости", possibleElements.map((row) => `${row.symbol}: ${formatInt(row.maxAtoms)}`)));
  root.append(sectionWithPills("Соединения", possibleCompounds.map((row) => `${row.formula}: ${formatInt(row.maxMolecules)}`)));
  root.append(sectionWithPills("Функциональные классы", possibleFunctions.map((row) => `${row.label}: ${formatInt(row.capacity)}`), "good"));
  root.append(sectionWithPills("Сборочные цели", possibleAssemblies.map((row) => `${row.label}: ${formatInt(row.capacity)}`), "good"));
  root.append(sectionWithPills("Активные листья", Object.entries(leafCounts).map(([leaf, count]) => `${leaf}: ${count}`), "warn"));

  els.summary.replaceChildren(root);
}

function renderQueryResult() {
  const result = activeQueryResult || {
    status: "empty",
    message: "Введите запрос слева: вода, глюкоза, проводник, техническое устройство."
  };
  const card = document.createElement("div");
  card.className = "query-card";
  const title = document.createElement("strong");
  title.textContent = result.status === "ok" ? result.target.label : statusLabel(result.status);
  card.append(title);

  if (result.status !== "ok") {
    const message = document.createElement("p");
    message.textContent = result.message;
    card.append(message);
    const options = result.candidates || result.suggestions || [];
    if (options.length) {
      card.append(pillList(options.slice(0, 8), result.status === "ambiguous" ? "warn" : ""));
    }
    els.queryResult.replaceChildren(card);
    return;
  }

  const verdict = document.createElement("span");
  if (result.structuralBlocked) {
    verdict.textContent = "Нельзя получить по текущим логическим дельтам rule-pack.";
  } else if (result.theoryVerdict && result.theoryVerdict.status === "possible") {
    verdict.textContent = result.capacity > 0
      ? `Возможно по дельтам; ресурсная емкость: ${formatInt(result.capacity)} ед.`
      : "Возможно по дельтам, но не хватает текущих ресурсов.";
  } else {
    verdict.textContent = result.capacity > 0
      ? `Можно получить: ${formatInt(result.capacity)} ед.`
      : "Нельзя получить из текущих ресурсов.";
  }
  card.append(verdict);
  card.append(pillList([
    `p+: ${formatInt(result.requirement.protons)}`,
    `n0: ${formatInt(result.requirement.neutrons)}`,
    `e-: ${formatInt(result.requirement.electrons)}`
  ]));

  if (result.capacity === 0) {
    card.append(pillList([
      `не хватает p+: ${formatInt(result.shortage.protons)}`,
      `не хватает n0: ${formatInt(result.shortage.neutrons)}`,
      `не хватает e-: ${formatInt(result.shortage.electrons)}`
    ], "warn"));
  }

  const definition = document.createElement("p");
  definition.textContent = result.definition;
  card.append(definition);
  if (result.theoryVerdict) {
    const reasons = result.theoryVerdict.reasons && result.theoryVerdict.reasons.length
      ? result.theoryVerdict.reasons
      : ["без запрещающих причин"];
    card.append(pillList([
      `дельта-статус: ${result.theoryVerdict.status}`,
      `origin: ${result.theoryVerdict.origin}`,
      ...reasons.slice(0, 5)
    ], result.theoryVerdict.status === "possible" ? "good" : "warn"));
  }
  card.append(pillList([`source: ${result.source}`, `rule-pack: ${result.rulePack}`, result.stateHash], "good"));
  if (result.chain) {
    card.append(pillList(result.chain));
  }
  if (result.classification) {
    card.append(pillList([
      `лист: ${result.classification.label}`,
      ...result.classification.path.map((step) => `${step.delta_id}=${step.value ? "true" : "false"}`)
    ], "warn"));
  }
  if (result.objects && result.objects.length) {
    card.append(pillList(result.objects.slice(0, 8), "good"));
  }

  els.queryResult.replaceChildren(card);
}

function statusLabel(status) {
  if (status === "unknown") {
    return "UNKNOWN";
  }
  if (status === "ambiguous") {
    return "Неоднозначный запрос";
  }
  return "Запрос";
}

function metric(label, value) {
  const node = document.createElement("div");
  node.className = "metric";
  const strong = document.createElement("strong");
  strong.textContent = value;
  const span = document.createElement("span");
  span.textContent = label;
  node.append(strong, span);
  return node;
}

function pillList(values, kind = "") {
  const list = document.createElement("div");
  list.className = "pill-list";
  values.forEach((value) => {
    const pill = document.createElement("span");
    pill.className = `pill ${kind}`;
    pill.textContent = value;
    list.append(pill);
  });
  return list;
}

function sectionWithPills(title, values, kind = "") {
  const section = document.createElement("div");
  const heading = document.createElement("h2");
  heading.textContent = title;
  const list = document.createElement("div");
  list.className = "pill-list";
  if (!values.length) {
    const empty = document.createElement("span");
    empty.className = "pill";
    empty.textContent = "нет";
    list.append(empty);
  } else {
    values.slice(0, 14).forEach((value) => {
      const pill = document.createElement("span");
      pill.className = `pill ${kind}`;
      pill.textContent = value;
      list.append(pill);
    });
  }
  section.append(heading, list);
  return section;
}

function renderEventLog() {
  const items = eventItems.map((item) => {
    const li = document.createElement("li");
    const strong = document.createElement("strong");
    strong.textContent = `${item.stage}: `;
    const text = document.createTextNode(item.message);
    li.append(strong, text);
    return li;
  });
  els.eventLog.replaceChildren(...items);
}

function renderClassifications() {
  const rows = currentState.classifications.map((row) => {
    const item = document.createElement("div");
    item.className = "class-row";
    const title = document.createElement("strong");
    title.textContent = `${row.label} -> ${row.classification.label}`;
    const count = document.createElement("span");
    count.textContent = `емкость: ${formatInt(row.count)}; тип: ${row.kind}`;
    const path = document.createElement("span");
    path.textContent = row.classification.path
      .map((step) => `${step.delta_id}=${step.value ? "true" : "false"}`)
      .join(" -> ");
    item.append(title, count, path);
    return item;
  });
  els.classificationTable.replaceChildren(...rows);
}

function renderCanvas() {
  const canvas = els.canvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(300, Math.floor(rect.height));
  if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  drawBackground(ctx, width, height);

  if (stageIndex === 0) {
    drawParticles(ctx, width, height, currentState);
  } else if (stageIndex === 1) {
    drawElements(ctx, width, height, currentState);
  } else if (stageIndex === 2) {
    drawCompounds(ctx, width, height, currentState);
  } else if (stageIndex === 3) {
    drawFunctions(ctx, width, height, currentState);
  } else {
    drawClassification(ctx, width, height, currentState);
  }
}

function drawBackground(ctx, width, height) {
  ctx.fillStyle = "#fbfcf8";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e2e7dc";
  ctx.lineWidth = 1;
  for (let x = 24; x < width; x += 48) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 24; y < height; y += 48) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawParticles(ctx, width, height, state) {
  const input = state.input;
  const entries = [
    { label: "p+", count: input.protons, color: "#c46a2b" },
    { label: "n0", count: input.neutrons, color: "#687174" },
    { label: "e-", count: input.electrons, color: "#445d9c" }
  ];
  const total = entries.reduce((sum, item) => sum + item.count, 0);
  const rng = createRng(total + input.volumeCells * 31);
  const maxDots = 360;
  const dots = [];
  entries.forEach((entry) => {
    const share = total === 0 ? 0 : Math.round((entry.count / total) * maxDots);
    for (let i = 0; i < share; i += 1) {
      dots.push(entry);
    }
  });
  dots.slice(0, maxDots).forEach((dot) => {
    const x = 18 + rng() * (width - 36);
    const y = 18 + rng() * (height - 36);
    ctx.fillStyle = dot.color;
    ctx.beginPath();
    ctx.arc(x, y, 3.4, 0, Math.PI * 2);
    ctx.fill();
  });
  drawLegend(ctx, entries.map((entry) => `${entry.label}: ${formatInt(entry.count)}`), 18, 24);
}

function drawElements(ctx, width, height, state) {
  const items = state.elements.filter((row) => row.maxAtoms > 0);
  if (!items.length) {
    drawEmpty(ctx, width, height, "Недостаточно частиц для выбранного диапазона элементов");
    return;
  }
  const columns = Math.max(2, Math.floor(width / 150));
  const cellW = width / columns;
  const cellH = 92;
  items.forEach((item, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const cx = col * cellW + cellW / 2;
    const cy = 64 + row * cellH;
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(cx, cy, 26, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "700 18px Segoe UI, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(item.symbol, cx, cy);
    ctx.fillStyle = "#1f2528";
    ctx.font = "12px Segoe UI, Arial";
    ctx.fillText(`${formatInt(item.maxAtoms)} атомов`, cx, cy + 42);
  });
}

function drawCompounds(ctx, width, height, state) {
  const items = state.compounds.filter((row) => row.maxMolecules > 0);
  if (!items.length) {
    drawEmpty(ctx, width, height, "Нет соединений с ненулевой емкостью");
    return;
  }
  const columns = Math.max(2, Math.floor(width / 180));
  const cellW = width / columns;
  const cellH = 88;
  items.forEach((item, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const x = col * cellW + 18;
    const y = 24 + row * cellH;
    const w = cellW - 36;
    ctx.fillStyle = compoundColor(item.family);
    roundRect(ctx, x, y, w, 58, 8);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "700 17px Segoe UI, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(item.formula, x + w / 2, y + 22);
    ctx.font = "12px Segoe UI, Arial";
    ctx.fillText(formatInt(item.maxMolecules), x + w / 2, y + 43);
  });
}

function drawFunctions(ctx, width, height, state) {
  const items = [
    ...state.functions.filter((row) => row.possible),
    ...state.assemblies.filter((row) => row.possible)
  ];
  if (!items.length) {
    drawEmpty(ctx, width, height, "Био- и техно-классы пока не выводятся");
    return;
  }
  const columns = width > 720 ? 2 : 1;
  const cellW = width / columns;
  items.forEach((item, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const x = col * cellW + 24;
    const y = 28 + row * 78;
    const w = cellW - 48;
    ctx.fillStyle = item.type && item.type.startsWith("bio") ? "#237a3b" : "#445d9c";
    roundRect(ctx, x, y, w, 54, 8);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "700 14px Segoe UI, Arial";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(item.label, x + 14, y + 21);
    ctx.font = "12px Segoe UI, Arial";
    ctx.fillText(`емкость ${formatInt(item.capacity)}`, x + 14, y + 39);
  });
}

function drawClassification(ctx, width, height, state) {
  const leafCounts = countBy(state.classifications, (row) => row.classification.label);
  const entries = Object.entries(leafCounts);
  if (!entries.length) {
    drawEmpty(ctx, width, height, "Нет объектов для классификации");
    return;
  }
  const max = Math.max(...entries.map(([, count]) => count));
  const left = 34;
  const top = 34;
  const rowH = 48;
  entries.forEach(([leaf, count], index) => {
    const y = top + index * rowH;
    const barW = Math.max(24, ((width - 240) * count) / max);
    ctx.fillStyle = leafColor(leaf);
    roundRect(ctx, left, y, barW, 24, 6);
    ctx.fill();
    ctx.fillStyle = "#1f2528";
    ctx.font = "13px Segoe UI, Arial";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`${leaf}: ${count}`, left + barW + 12, y + 12);
  });
  drawLegend(ctx, ["Доказательство уникальности: один boolean-выбор на каждой внутренней вершине"], 18, height - 24);
}

function drawEmpty(ctx, width, height, message) {
  ctx.fillStyle = "#687174";
  ctx.font = "15px Segoe UI, Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, width / 2, height / 2);
}

function drawLegend(ctx, lines, x, y) {
  ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
  const width = Math.min(520, Math.max(220, lines.join(" ").length * 6));
  roundRect(ctx, x - 8, y - 17, width, 30, 6);
  ctx.fill();
  ctx.fillStyle = "#1f2528";
  ctx.font = "12px Segoe UI, Arial";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(lines.join("   "), x, y);
}

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function compoundColor(family) {
  if (family === "bio") {
    return "#237a3b";
  }
  if (family === "tech") {
    return "#445d9c";
  }
  return "#1f7a68";
}

function leafColor(leaf) {
  if (leaf.includes("Поле")) {
    return "#7aa6b8";
  }
  if (leaf.includes("Свободные")) {
    return "#c46a2b";
  }
  if (leaf.includes("Твердое")) {
    return "#596066";
  }
  if (leaf.includes("Жидкость")) {
    return "#1f7a68";
  }
  if (leaf.includes("Плазма")) {
    return "#9a5b10";
  }
  return "#445d9c";
}

function countBy(items, getKey) {
  return items.reduce((acc, item) => {
    const key = getKey(item);
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
}

function createRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function formatInt(value) {
  return new Intl.NumberFormat("ru-RU").format(value);
}
