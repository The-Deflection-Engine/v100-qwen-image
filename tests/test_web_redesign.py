"""The redesigned Create page: no inline styles, no lost ids, and the new
interactions EXECUTED in node.

Three things a string check cannot see, so the last section extracts the real
functions from app.js verbatim and drives them over a DOM built from the real
index.html markup:

* the 1K/2K tabs must follow the selected preset by ANY route (a tile,
  Random, an import, an "Apply ratio" hint) and must not offer a tier the
  server withheld from a remote account;
* the character counter must agree with the same Python-strip semantics the
  "prompt is required" validation uses;
* an info popover must open on click, close on a click outside, and close on
  Escape with focus returned to the button that opened it.

The static half enforces the two structural promises of this redesign: the
Content-Security-Policy forbids inline styles and scripts, so no ``style="``
attribute may appear, and every id that existed before must still exist.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "qwen_engine" / "web_static"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _read(name: str) -> str:
    path = WEB_DIR / name
    assert path.is_file(), f"missing packaged static asset: {path}"
    return path.read_text(encoding="utf-8")


def _js_function(js: str, signature: str) -> str:
    m = re.search(rf"function {re.escape(signature)}\([\s\S]*?\n  \}}", js)
    assert m, f"{signature} not found in app.js"
    return m.group(0)


def _js_var(js: str, name: str) -> str:
    """One top-level ``var <name> = ...;`` statement from app.js, verbatim.

    Found by scanning to the statement's terminator, because the value wraps
    over many lines and may carry a trailing comment.
    """
    start = re.search(rf"^  var {re.escape(name)} =\s", js, re.M)
    assert start, f"{name} not found in app.js"
    depth = 0
    quote = ""
    index = start.end()
    while index < len(js):
        char = js[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = ""
        elif char in "'\"`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ";" and depth == 0:
            return js[start.start():index + 1]
        index += 1
    raise AssertionError(f"{name} has no statement end in app.js")


# ------------------------------------------------------------- static: CSP


def test_the_page_carries_no_inline_style_or_script():
    """The server sends style-src 'self'; script-src 'self'.

    An inline attribute would be silently dropped by the browser, so every
    rule belongs in app.css. The only script element is the deferred file.
    """
    html = _read("index.html")
    assert "style=" not in html.replace("stylesheet", "")
    assert not re.search(r"<style[\s>]", html)
    scripts = re.findall(r"<script[^>]*>", html)
    assert scripts, "the page loads its script"
    for tag in scripts:
        assert re.search(r'src="[^"/]+\.js"', tag), tag
        assert "defer" in tag, tag


def test_the_stylesheet_is_self_contained():
    css = _read("app.css")
    assert "@import" not in css
    assert not re.search(r"url\(\s*['\"]?https?:", css)
    # The hidden rule stays first: it must win over every component display.
    stripped = re.sub(r"/\*[\s\S]*?\*/", "", css)
    first_rule = re.search(r"([^\s][^{]*)\{", stripped)
    assert first_rule and first_rule.group(1).strip() == "[hidden]", \
        "[hidden] { display: none !important; } must stay the first rule"


# --------------------------------------------------------- static: ids kept


def test_every_pre_redesign_id_survives():
    """Moving an element is fine; dropping its id breaks the script and tests."""
    before = {
        line.strip()
        for line in (FIXTURES / "web_ids_before_redesign.txt").read_text(
            encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert len(before) > 250, "the fixture must list the whole pre-redesign page"
    now = set(re.findall(r'id="([^"]+)"', _read("index.html")))
    assert not (before - now), sorted(before - now)


def test_every_id_the_script_looks_up_exists_in_the_markup():
    """An id app.js reads but the page never ships is a dead control.

    Template ids (``'model-size-' + id``) are built at runtime, so they are
    matched by prefix rather than demanded literally.
    """
    html = _read("index.html")
    js = _read("app.js")
    ids = set(re.findall(r'id="([^"]+)"', html))
    literal = set(re.findall(r"\$\('([a-z0-9-]+)'\)", js))
    literal |= set(re.findall(r"setText\('([a-z0-9-]+)'", js))
    literal |= set(re.findall(r"hideIf\('([a-z0-9-]+)'", js))
    # ``setText('model-size-' + id)`` is a template, not a literal id: its
    # prefix is only dynamic when the script really concatenates onto it.
    missing = sorted(name for name in literal
                     if name not in ids and f"'{name}' +" not in js)
    assert not missing, missing


def test_the_new_design_elements_are_in_the_markup():
    html = _read("index.html")
    for node in ("prompt-char-count", "preset-tier-tabs", "preset-tier-1k",
                 "preset-tier-2k", "presets-1k", "presets-2k-group",
                 "presets-2k"):
        assert f'id="{node}"' in html, node
    # One "?" per static explanation, each pointing at the help id it opens, so
    # aria-describedby on the control still resolves.
    for help_id in ("prompt-help", "preset-help", "edit-mode-help",
                    "transparency-help", "seed-help", "recipe-help"):
        assert f'id="{help_id}"' in html, help_id
    buttons = re.findall(r'<button[^>]*class="info-btn"[^>]*>', html)
    assert len(buttons) == 6, buttons
    for tag in buttons:
        assert 'aria-expanded="false"' in tag, tag
        assert "aria-controls=" in tag, tag


def test_the_tier_tabs_and_groups_are_marked_up_for_the_script():
    html = _read("index.html")
    tabs = re.search(r'<span class="preset-tier-tabs" id="preset-tier-tabs"'
                     r'[\s\S]*?</span>', html).group(0)
    assert 'role="tablist"' in tabs
    assert tabs.count('role="tab"') == 2
    assert 'data-tier="1K"' in tabs and 'data-tier="2K"' in tabs
    assert 'aria-selected="true"' in tabs, "one tier starts selected"
    # The 2K grid lives inside the group whose hidden flag the server sets.
    group = re.search(r'<div id="presets-2k-group"[\s\S]*?</div>\s*</fieldset>', html)
    assert group and 'id="presets-2k"' in group.group(0)
    assert 'id="presets-1k"' in html


# ----------------------------------------------------- executed in node


HARNESS_PARTS = (
    "$", "setText", "pyStrip",
    "presetNameFor", "presetTierAllowed", "syncPresetTier",
    "syncPresetButtons", "updatePromptCharCount",
    "infoPopoverTarget", "setInfoPopover", "clearInfoPopoverDismissal",
    "closeInfoPopovers", "toggleInfoPopover",
    "visibleBand", "placeInfoPopover", "closeInfoPopoversOnEscape",
)
HARNESS_VARS = ("PRESETS",)

_HARNESS = """// Built by tests/test_web_redesign.py: the real app.js functions, extracted
// verbatim, driven over a DOM assembled from the ids index.html really ships.
// Nothing here reimplements a product rule.
var fs = require("node:fs");
__PARTS__

// A DOM thin enough to execute the real functions and thick enough for the
// geometry placeInfoPopover measures: rects that DEPEND on the placement (the
// panel's rect follows the flip class, the height cap and the horizontal
// offset), a class list, a style object with setProperty, and a parent chain
// up to a clipping, scrolling pane.
var RECTS = {};
var NODES = {};
function classList(owner) {
  return {
    contains: function (name) { return owner.classes.indexOf(name) >= 0; },
    add: function (name) {
      if (owner.classes.indexOf(name) < 0) owner.classes.push(name);
    },
    remove: function (name) {
      var at = owner.classes.indexOf(name);
      if (at >= 0) owner.classes.splice(at, 1);
    }
  };
}
// A style object: plain assignments (panel.style.left = "12px") and the two
// CSSOM calls the placement uses must both be visible to the next read.
function styleObject() {
  var values = {};
  function get(target, key) {
    if (key === "setProperty") {
      return function (name, value) { values[name] = String(value); };
    }
    if (key === "removeProperty") {
      return function (name) { delete values[name]; };
    }
    if (key === "getPropertyValue") {
      return function (name) { return values[name] || ""; };
    }
    return target[key];
  }
  return new Proxy({}, {
    get: function (target, key) { return get(target, key); },
    set: function (target, key, value) {
      if (value === "" || value === null || value === undefined) delete target[key];
      else target[key] = value;
      return true;
    }
  });
}
function node(id, classes) {
  var self = { id: id, textContent: "", hidden: false, dataset: {}, attrs: {},
               style: styleObject(), classes: classes || [], parentNode: null,
               nodeType: 1, focused: false, offsetHeight: 0,
               setAttribute: function (name, value) { this.attrs[name] = value; },
               getAttribute: function (name) {
                 return Object.prototype.hasOwnProperty.call(this.attrs, name)
                   ? this.attrs[name] : null;
               },
               hasAttribute: function (name) {
                 return Object.prototype.hasOwnProperty.call(this.attrs, name);
               },
               removeAttribute: function (name) { delete this.attrs[name]; },
               getBoundingClientRect: function () {
                 if (self.panelOf) return panelRect(self);
                 var r = RECTS[id] || { top: 0, bottom: 0, left: 0, right: 0,
                                        width: 0, height: 0 };
                 return r;
               },
               focus: function () { this.focused = true; } };
  self.classList = classList(self);
  return self;
}
// A panel's rect follows its own placement, like the browser's: the CSS
// constant puts it GAP below the wrapper, but the real distance measured in
// Chrome is six pixels more (the wrapper's line box), which is exactly why
// placeInfoPopover must decide from the measured rect instead of the constant.
var GAP = 8;
var LINE_BOX = 6;
function panelRect(panel) {
  var wrapper = panel.parentNode;
  var anchor = wrapper.getBoundingClientRect();
  var height = panel.naturalHeight;
  var cap = parseFloat(panel.style.maxHeight);
  if (cap > 0 && cap < height) height = Math.floor(cap);
  var flipped = wrapper.classList.contains("flip-up");
  var gap = GAP + LINE_BOX;
  var top = flipped ? anchor.top - gap - height : anchor.bottom + gap;
  var left = anchor.left + (parseFloat(panel.style.left) || 0);
  var width = panel.naturalWidth;
  return { top: top, bottom: top + height, left: left,
           right: left + width, width: width, height: height };
}

// The scrolling settings pane every popover sits inside, and the window.
var PANE = node("pane");
RECTS["pane"] = { top: 100, bottom: 700, left: 0, right: 400,
                  width: 400, height: 600 };
var window = {
  innerHeight: 900, innerWidth: 1440,
  getComputedStyle: function (target) {
    // The pane both scrolls and clips, so it bounds BOTH axes.
    if (target === PANE) return { overflowY: "auto", overflowX: "auto" };
    return { overflowY: "visible", overflowX: "visible" };
  }
};
function ensure(id) {
  if (!NODES[id]) NODES[id] = node(id);
  return NODES[id];
}
// The ids the functions under test read, with the starting state index.html
// gives them: every button starts collapsed, one tier starts selected.
["prompt", "prompt-char-count", "width", "height",
 "preset-tier-tabs", "preset-tier-1k", "preset-tier-2k",
 "presets-2k-group", "prompts-1k", "presets-1k", "presets-2k"
].forEach(function (id) { ensure(id); });
NODES["preset-tier-1k"].attrs["aria-selected"] = "true";
NODES["preset-tier-2k"].attrs["aria-selected"] = "false";

var BUTTONS = [];
var HELP_IDS = ["prompt-help", "preset-help", "edit-mode-help",
                "transparency-help", "seed-help", "recipe-help"];
HELP_IDS.forEach(function (helpId, index) {
  var wrapper = node("wrap-" + index, ["info-popover-wrapper"]);
  wrapper.parentNode = PANE;
  var button = node("btn-" + index);
  button.attrs["aria-controls"] = helpId;
  button.attrs["aria-expanded"] = "false";
  button.parentNode = wrapper;
  BUTTONS.push(button);
  var panel = ensure(helpId);
  panel.offsetHeight = 140;
  panel.naturalHeight = 140;
  panel.naturalWidth = 300;
  panel.panelOf = true;
  panel.parentNode = wrapper;
  // Every button sits 200px below the pane's top, well inside it, so no panel
  // needs flipping unless a case says so.
  RECTS["btn-" + index] = { top: 300, bottom: 318, left: 0, right: 18,
                            width: 18, height: 18 };
  RECTS["wrap-" + index] = RECTS["btn-" + index];
});

var document = {
  getElementById: function (id) { return NODES[id] || null; },
  querySelectorAll: function (selector) {
    if (selector === ".preset-btn") return [];
    if (selector === ".info-btn") return BUTTONS;
    return [];
  },
  querySelector: function (selector) {
    if (selector !== '.info-btn[aria-expanded="true"]') return null;
    for (var i = 0; i < BUTTONS.length; i++) {
      if (BUTTONS[i].attrs["aria-expanded"] === "true") return BUTTONS[i];
    }
    return null;
  }
};

function tierState() {
  return {
    shown: presetTierShown,
    selected: [NODES["preset-tier-1k"].attrs["aria-selected"],
               NODES["preset-tier-2k"].attrs["aria-selected"]],
    hidden: { tabs: NODES["preset-tier-tabs"].hidden,
              oneK: NODES["presets-1k"].hidden,
              twoK: NODES["presets-2k"].hidden }
  };
}

var cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
var out = cases.map(function (step) {
  if (step.kind === "tier") {
    // A 2K permission withdrawn or granted, then a route that selects a preset.
    NODES["presets-2k-group"].hidden = step.denied;
    NODES["width"].value = String(step.width);
    NODES["height"].value = String(step.height);
    if (step.tab) syncPresetTier(step.tab);
    syncPresetButtons();
    return { kind: step.kind, state: tierState() };
  }
  if (step.kind === "count") {
    NODES["prompt"].value = step.text;
    updatePromptCharCount();
    return { kind: step.kind, text: NODES["prompt-char-count"].textContent };
  }
  if (step.kind === "popover") {
    BUTTONS.forEach(function (button) {
      button.attrs["aria-expanded"] = "false";
      button.focused = false;
      button.parentNode.classList.remove("flip-up");
    });
    HELP_IDS.forEach(function (id) {
      NODES[id].removeAttribute("data-open");
      NODES[id].removeAttribute("data-closed");
      NODES[id].style.left = "";
      NODES[id].style.maxHeight = "";
    });
    if (step.nearBottom) {
      // 60px of pane left below the button, a 140px panel: it cannot fit.
      RECTS["btn-" + step.toggle] = { top: 622, bottom: 640, left: 0, right: 18,
                                      width: 18, height: 18 };
      RECTS["wrap-" + step.toggle] = RECTS["btn-" + step.toggle];
    }
    if (step.nearRight) {
      // A 300px panel opening from a button 52px from the pane's right edge:
      // opened straight down it would hang 248px outside the clip.
      RECTS["btn-" + step.toggle] = { top: 300, bottom: 318, left: 330,
                                      right: 348, width: 18, height: 18 };
      RECTS["wrap-" + step.toggle] = RECTS["btn-" + step.toggle];
    }
    if (step.shortPane) {
      // A 180px-tall pane with the button in the middle: 81px of room above,
      // so the cap binds BELOW any constant floor (the buggy 120px one pushed
      // the panel 39px back out over the pane's top edge).
      RECTS["pane"] = { top: 100, bottom: 280, left: 0, right: 400,
                        width: 400, height: 180 };
      RECTS["btn-" + step.toggle] = { top: 195, bottom: 213, left: 0, right: 18,
                                      width: 18, height: 18 };
      RECTS["wrap-" + step.toggle] = RECTS["btn-" + step.toggle];
      NODES[HELP_IDS[step.toggle]].naturalHeight = 400;
      NODES[HELP_IDS[step.toggle]].offsetHeight = 400;
    }
    if (step.tallPanel) {
      // A 700px panel in a 600px pane: no direction fits it, so the height must
      // be capped to the room measured from the anchor. A constant floor above
      // that room (the 120px one this bug had) puts it back outside the edge.
      RECTS["btn-" + step.toggle] = { top: 400, bottom: 418, left: 0, right: 18,
                                      width: 18, height: 18 };
      RECTS["wrap-" + step.toggle] = RECTS["btn-" + step.toggle];
      NODES[HELP_IDS[step.toggle]].naturalHeight = 700;
      NODES[HELP_IDS[step.toggle]].offsetHeight = 700;
    }
    toggleInfoPopover(BUTTONS[step.toggle]);
    var opened = {};
    HELP_IDS.forEach(function (id) {
      opened[id] = NODES[id].getAttribute("data-open");
    });
    var expanded = BUTTONS.map(function (b) {
      return b.attrs["aria-expanded"];
    });
    // A second click on the same button is a close.
    if (step.again) toggleInfoPopover(BUTTONS[step.toggle]);
    if (step.escape) closeInfoPopoversOnEscape();
    else if (step.outside) closeInfoPopovers(null);
    var after = {};
    HELP_IDS.forEach(function (id) {
      after[id] = NODES[id].getAttribute("data-open");
    });
    return {
      kind: step.kind,
      opened: opened,
      expanded: expanded,
      after: after,
      afterExpanded: BUTTONS.map(function (b) { return b.attrs["aria-expanded"]; }),
      afterClosed: HELP_IDS.map(function (id) {
        return NODES[id].getAttribute("data-closed");
      }),
      // What the placement wrote, so a case can assert containment.
      placed: HELP_IDS.map(function (id) {
        var panel = NODES[id];
        return { id: id, left: panel.style.left || null,
                 maxHeight: panel.style.maxHeight || null,
                 arrow: panel.style.getPropertyValue("--arrow-x") || null,
                 rect: panel.getBoundingClientRect() };
      }),
      band: (function () {
        var top = 0, bottom = window.innerHeight;
        var left = 0, right = window.innerWidth;
        for (var n = PANE; n && n.nodeType === 1; n = n.parentNode) {
          var st = window.getComputedStyle(n);
          var b = n.getBoundingClientRect();
          if (/(auto|scroll|hidden|clip)/.test(st.overflowY)) {
            top = Math.max(top, b.top); bottom = Math.min(bottom, b.bottom);
          }
          if (/(auto|scroll|hidden|clip)/.test(st.overflowX)) {
            left = Math.max(left, b.left); right = Math.min(right, b.right);
          }
        }
        return { top: top, bottom: bottom, left: left, right: right };
      })(),
      flipped: BUTTONS.map(function (b) {
        return b.parentNode.classList.contains("flip-up");
      }),
      focused: BUTTONS.map(function (b) { return !!b.focused; })
    };
  }
  throw new Error("unknown step " + step.kind);
});
process.stdout.write(JSON.stringify(out));
"""


def _run_harness(tmp_path: Path, cases: list[dict]) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read("app.js")
    parts = "\n".join([_js_var(js, name) for name in HARNESS_VARS]
                      + [_js_function(js, fn) for fn in HARNESS_PARTS])
    # presetTierShown is a module-level var the functions read and write.
    parts += "\n" + _js_var(js, "presetTierShown")
    harness = tmp_path / "redesign_harness.js"
    payload = tmp_path / "cases.json"
    harness.write_text(_HARNESS.replace("__PARTS__", parts), encoding="utf-8")
    payload.write_text(json.dumps(cases), encoding="utf-8")
    run = subprocess.run([node, str(harness), str(payload)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:2000]}"
    return json.loads(run.stdout)


def test_the_tier_tab_follows_the_selected_preset(tmp_path):
    """Selecting a 2K size by ANY route must reveal the 2K tier.

    Random, an imported plan and an "Apply ratio" hint all end in the same
    syncPresetButtons call as a tile click, so the route is not what is
    tested — the derived state is.
    """
    cases = [
        {"kind": "tier", "denied": False, "width": 1024, "height": 1024},
        {"kind": "tier", "denied": False, "width": 2048, "height": 2048},
        {"kind": "tier", "denied": False, "width": 1536, "height": 2752},
        {"kind": "tier", "denied": False, "width": 1152, "height": 864},
    ]
    out = _run_harness(tmp_path, cases)
    square, wide, tall, back = (item["state"] for item in out)
    assert square["shown"] == "1K"
    assert square["selected"] == ["true", "false"]
    assert square["hidden"]["oneK"] is False and square["hidden"]["twoK"] is True
    for state in (wide, tall):
        assert state["shown"] == "2K", state
        assert state["selected"] == ["false", "true"], state
        assert state["hidden"]["oneK"] is True and state["hidden"]["twoK"] is False
    # Back to a 1K preset: the tabs follow again, without any manual click.
    assert back["shown"] == "1K" and back["selected"] == ["true", "false"]


def test_a_hand_typed_size_keeps_the_tier_on_screen(tmp_path):
    """No exact match means the tiles must not jump under the pointer."""
    cases = [
        {"kind": "tier", "denied": False, "width": 2048, "height": 2048},
        {"kind": "tier", "denied": False, "width": 1200, "height": 800},
    ]
    out = _run_harness(tmp_path, cases)
    assert out[0]["state"]["shown"] == "2K"
    assert out[1]["state"]["shown"] == "2K", "an unmatched size keeps the last tier"
    assert out[1]["state"]["selected"] == ["false", "true"]


def test_a_withheld_tier_cannot_be_shown_or_selected(tmp_path):
    """A remote account without 2K must not reach the 2K grid at all."""
    cases = [
        {"kind": "tier", "denied": True, "width": 1024, "height": 1024},
        {"kind": "tier", "denied": True, "width": 1024, "height": 1024,
         "tab": "2K"},
        {"kind": "tier", "denied": True, "width": 2048, "height": 2048},
    ]
    out = _run_harness(tmp_path, cases)
    for item in out:
        state = item["state"]
        assert state["hidden"]["tabs"] is True, "the tab pair goes with the group"
        assert state["hidden"]["twoK"] is True
        assert state["hidden"]["oneK"] is False
        assert state["shown"] == "1K"
        assert state["selected"] == ["true", "false"]


def test_the_tier_tabs_filter_without_writing_a_dimension(tmp_path):
    """A tab is a view filter: pressing it must not touch the draft."""
    js = _read("app.js")
    body = _js_function(js, "syncPresetTier")
    for control in ("width", "height", "seed", "vae-tiling", "recipe"):
        assert f"$('{control}')" not in body, f"a tab must not write #{control}"
    # ...and it never writes the group's own hidden flag, which is the
    # server's permission answer, only the tier grids inside it.
    assert "presets-2k-group').hidden =" not in body
    assert 'presets-1k' in body and "'presets-2k'" in body or "$('presets-2k')" in body


def test_the_character_counter_uses_the_validation_semantics(tmp_path):
    """The counter and the "prompt is required" error must never disagree."""
    cases = [
        {"kind": "count", "text": ""},
        {"kind": "count", "text": "a"},
        {"kind": "count", "text": "  padded  "},
        {"kind": "count", "text": "   "},
        {"kind": "count", "text": "two words"},
        {"kind": "count", "text": "m\u00fcnchen \u2014 ok"},
    ]
    out = _run_harness(tmp_path, cases)
    texts = [item["text"] for item in out]
    assert texts == ["0 characters", "1 character", "6 characters",
                     "0 characters", "9 characters", "12 characters"]
    # It is called from the one place every text change funnels through.
    js = _read("app.js")
    assert "updatePromptCharCount();" in _js_function(js, "bumpPromptRevision")
    # ...and once at load, so a restored draft shows a real number.
    init = js[js.index("function init() {"):js.index("\n  }", js.index("function init() {"))]
    assert "updatePromptCharCount();" in init


def test_a_popover_opens_on_click_and_closes_on_a_click_outside(tmp_path):
    cases = [{"kind": "popover", "toggle": 0, "outside": True}]
    out = _run_harness(tmp_path, cases)[0]
    assert out["opened"]["prompt-help"] == "true"
    assert out["opened"]["seed-help"] is None, "one panel at a time"
    assert out["expanded"][0] == "true"
    assert out["after"]["prompt-help"] is None, "an outside click closes it"
    assert out["afterExpanded"][0] == "false"


def test_opening_a_second_popover_closes_the_first(tmp_path):
    js = _read("app.js")
    body = _js_function(js, "toggleInfoPopover")
    assert "closeInfoPopovers(button);" in body, "the rest are closed first"
    assert "aria-expanded" in body
    # Clicking the open button again closes it, rather than re-opening it.
    cases = [{"kind": "popover", "toggle": 2, "again": True}]
    out = _run_harness(tmp_path, cases)[0]
    assert out["opened"]["edit-mode-help"] == "true"
    assert out["after"]["edit-mode-help"] is None
    assert out["afterExpanded"][2] == "false"


def test_escape_closes_the_popover_and_restores_focus(tmp_path):
    cases = [{"kind": "popover", "toggle": 4, "escape": True}]
    out = _run_harness(tmp_path, cases)[0]
    assert out["opened"]["seed-help"] == "true"
    assert out["after"]["seed-help"] is None
    assert out["afterExpanded"][4] == "false"
    assert out["focused"][4] is True, "focus returns to the opener"
    assert not any(out["focused"][i] for i in (0, 1, 2, 3, 5))
    # The page wires Escape to exactly that function, and hover/focus remain
    # CSS so a mouse or keyboard user sees the panel without clicking.
    js = _read("app.js")
    handler = js[js.index("document.addEventListener('keydown'"):]
    handler = handler[:handler.index("});")]
    assert "closeInfoPopoversOnEscape();" in handler
    css = _read("app.css")
    assert ".info-popover-wrapper:hover .info-popover" in css
    assert ".info-popover-wrapper:focus-within .info-popover" in css
    assert '.info-popover[data-open="true"]' in css
    # Escape returns focus to the button, which lives INSIDE the wrapper, so
    # :focus-within would immediately re-open what it just closed. The pin the
    # script sets must outrank both convenience reveals, i.e. come after them.
    pin = css.index('.info-popover[data-closed="true"]')
    assert pin > css.index(".info-popover-wrapper:hover .info-popover")
    assert pin > css.index(".info-popover-wrapper:focus-within .info-popover")
    assert out["afterClosed"][4] == "true", "Escape pins the panel shut"


def test_a_dismissed_popover_reopens_only_on_a_new_interaction(tmp_path):
    """A pinned panel must forget the pin when the user acts on the button again."""
    cases = [
        {"kind": "popover", "toggle": 4, "escape": True},
        {"kind": "popover", "toggle": 4},
    ]
    first, second = _run_harness(tmp_path, cases)
    assert first["afterClosed"][4] == "true"
    assert second["opened"]["seed-help"] == "true", "a fresh click still opens it"
    js = _read("app.js")
    # A focus event must NOT clear the pin: Escape itself moves focus there.
    focus_handler = js[js.index("button.addEventListener('focus'"):]
    focus_handler = focus_handler[:focus_handler.index("});")]
    assert "clearInfoPopoverDismissal" not in focus_handler
    pointer_handler = js[js.index("button.addEventListener('pointerenter'"):]
    pointer_handler = pointer_handler[:pointer_handler.index("});")]
    assert "clearInfoPopoverDismissal(button);" in pointer_handler


def test_a_popover_near_the_pane_edge_is_kept_inside_on_both_axes(tmp_path):
    """The clipping pane clips sideways too, and a short pane caps the height.

    A 300px panel opening from a "?" near the sidebar's right edge hung out of
    the pane's clipped side, where no scrolling could reach it; and a panel
    opened inside a 226px-tall pane reached past the pane's top because the
    height cap used a constant floor instead of the room measured from the
    anchor. Both are geometry, so both are asserted on the placed rect against
    the visible band.
    """
    cases = [
        {"kind": "popover", "toggle": 3, "nearRight": True},
        {"kind": "popover", "toggle": 1, "tallPanel": True},
        {"kind": "popover", "toggle": 0, "shortPane": True},
    ]
    right, tall, short = _run_harness(tmp_path, cases)

    def assert_inside(item, index):
        band, rect = item["band"], item["placed"][index]["rect"]
        assert rect["left"] >= band["left"], (rect, band)
        assert rect["right"] <= band["right"], (rect, band)
        assert rect["top"] >= band["top"], (rect, band)
        assert rect["bottom"] <= band["bottom"], (rect, band)

    assert_inside(right, 3)
    # The panel is slid left of its button, and the arrow moves with it so the
    # tail still points at the control instead of floating free.
    placed = right["placed"][3]
    assert placed["left"] is not None and placed["left"].startswith("-")
    assert int(placed["arrow"].rstrip("px")) > 12, placed["arrow"]

    assert tall["flipped"][1] is True, "it cannot fit below"
    assert_inside(tall, 1)
    # Exactly the measured room between the flipped panel's bottom edge and the
    # pane's top (386 - 100), never a constant that ignores it.
    assert tall["placed"][1]["maxHeight"] == "286px", tall["placed"][1]

    # A room SMALLER than any sensible floor: the cap is the room itself.
    assert short["flipped"][0] is True
    assert_inside(short, 0)
    assert short["placed"][0]["maxHeight"] == "81px", short["placed"][0]

    # A panel revealed by hover or focus alone must open at its stylesheet
    # position, so closing drops every placement the script wrote.
    close = _js_function(_read("app.js"), "setInfoPopover")
    assert "removeProperty('max-height')" in close
    assert "removeProperty('left')" in close


def test_a_popover_near_the_pane_bottom_opens_upward(tmp_path):
    """The one clip scrolling cannot fix: a panel below a pane's bottom edge.

    The sidebar and the settings pane both scroll and both clip, so a panel
    that does not fit below its button must open upward instead.
    """
    cases = [{"kind": "popover", "toggle": 5, "nearBottom": True}]
    out = _run_harness(tmp_path, cases)[0]
    assert out["opened"]["recipe-help"] == "true"
    assert out["flipped"][5] is True
    assert not any(out["flipped"][i] for i in range(5)), "only the tight one flips"
    placed = out["placed"][5]
    assert placed["rect"]["top"] >= out["band"]["top"], (placed, out["band"])
    css = _read("app.css")
    assert ".info-popover-wrapper.flip-up .info-popover {" in css
    assert "bottom: calc(100% + 8px);" in css
    assert "max-height: min(60vh, 420px); overflow-y: auto;" in css
    # Flipping alone is not enough if a card clips: the panel of a field in a
    # settings card was painted entirely outside the card and so invisible. A
    # card must be rounded, never a clip; its header rounds its own corners.
    card = css[css.index(".control-card {"):]
    card = card[:card.index("}")]
    assert "overflow: hidden" not in card, card
    header = css[css.index(".card-header {"):]
    header = header[:header.index("}")]
    assert "border-radius: calc(var(--radius) - 1px) calc(var(--radius) - 1px) 0 0;" in header


def test_the_popover_panel_cannot_be_clipped_by_the_sidebar(tmp_path):
    """The sidebar scrolls, so a panel must not be positioned outside it."""
    css = _read("app.css")
    rule = re.search(r"\.info-popover \{([^}]*)\}", css)
    assert rule and "position: absolute" in rule.group(1)
    assert "width: min(300px, calc(100vw - 48px))" in rule.group(1), \
        "never wider than the viewport it may be shown in"
    assert "z-index: 100" in rule.group(1)


def test_every_nav_link_routes_to_a_real_view():
    """A nav link whose hash is missing from VIEWS is a dead tab.

    #nav-help shipped with #view-help, but VIEWS omitted "help", so clicking
    Help kept Settings on screen. Every nav link is checked against the map,
    and every heading in the map must be focusable in the markup.
    """
    js = _read("app.js")
    html = _read("index.html")
    views = _js_var(js, "VIEWS")
    headings = _js_var(js, "VIEW_HEADINGS")
    links = re.findall(r'<a href="#(\w+)" id="nav-\w+"[^>]*>', html)
    assert sorted(links) == ["admin", "create", "help", "history", "settings"]
    for name in links:
        # The map is quoted with single quotes in app.js, so match either.
        entry = re.search(rf"\b{name}:\s*['\"]([\w-]+)['\"]", views)
        assert entry, f"#nav-{name} has no view in VIEWS"
        view_id = entry.group(1)
        assert f'id="{view_id}"' in html, f"{view_id} is not in the markup"
        heading = re.search(rf"\b{name}:\s*['\"]([\w-]+)['\"]", headings)
        assert heading, f"#nav-{name} has no heading"
        heading_id = heading.group(1)
        tag = re.search(rf'<h2[^>]*id="{heading_id}"[^>]*>', html)
        assert tag and 'tabindex="-1"' in tag.group(0), heading_id


def test_reduced_motion_disables_the_popover_and_chip_animation():
    css = _read("app.css")
    block = css[css.index("@media (prefers-reduced-motion: reduce)"):]
    assert "animation-duration: 0.001ms !important" in block
    assert "transition-duration: 0.001ms !important" in block
    assert ".connection-chip.is-ready::after { display: none; }" in block


def test_the_connection_chip_draws_its_dot_in_css():
    """The script only writes the chip's text, so the light must be a ::before."""
    css = _read("app.css")
    dot = re.search(r"\.connection-chip::before \{([^}]*)\}", css)
    assert dot and "border-radius: 50%" in dot.group(1)
    ready = re.search(r"\.connection-chip\.is-ready::before \{([^}]*)\}", css)
    assert ready and "var(--ok)" in ready.group(1)
    pulse = re.search(r"\.connection-chip\.is-ready::after \{([^}]*)\}", css)
    assert pulse and "animation:" in pulse.group(1)
