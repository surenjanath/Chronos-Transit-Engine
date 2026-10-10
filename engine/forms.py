"""Server-side validation for every operator input."""
import uuid

from django import forms

from cte import ingest
from cte.config import DEFAULT_PIPELINE, UPLOAD_DIR

# key, label, help, min, max, integer?
PIPELINE_FIELDS = [
    ("sigma_k", r"Sigma-clip threshold \(k\)", r"Reject \(|\tilde F - \mu_{\mathrm{roll}}| > k\,\sigma_{\mathrm{roll}}\)", 2, 20, False),
    ("clip_window", "Sigma-clip window", "Rolling window, cadences (odd)", 11, 2001, True),
    ("detrend_window_days", "Detrend window", "Savitzky-Golay length, days", 0.2, 10, False),
    ("detrend_polyorder", "Detrend polynomial order", r"PRD default \(p = 3\)", 1, 5, True),
    ("period_min", "Minimum period", "BLS lower bound, days", 0.2, 50, False),
    ("period_max", "Maximum period", "Capped at half the baseline", 0.5, 100, False),
    ("q_min", r"Minimum duty cycle \(q\)", r"\(q = T_{\mathrm{dur}}/P\) lower bound", 0.002, 0.1, False),
    ("q_max", r"Maximum duty cycle \(q\)", r"\(q = T_{\mathrm{dur}}/P\) upper bound", 0.02, 0.3, False),
    ("bls_oversample", "BLS oversampling", "Frequency-grid density", 1, 10, True),
    ("impact_param", r"Impact parameter \(b\)", "Assumed for duration", 0, 1.19, False),
    ("bond_albedo", r"Bond albedo \(A_B\)", "For equilibrium temperature", 0, 0.99, False),
    ("sde_threshold", "SDE flag threshold", "Flag weaker BLS peaks", 0, 50, False),
    ("planet_confidence", "Queue confidence", r"Minimum \(P_{\mathrm{planet}}\) to auto-queue", 0.34, 1, False),
]


def _pipeline_field(spec, required):
    key, label, help_text, lo, hi, is_int = spec
    cls = forms.IntegerField if is_int else forms.FloatField
    field = cls(label=label, help_text=help_text, min_value=lo, max_value=hi, required=required)
    field.widget.attrs.update({"step": "any", "placeholder": str(DEFAULT_PIPELINE[key])})
    return field


class PipelineMixin:
    prefix_key = ""

    def add_pipeline_fields(self, required: bool):
        for spec in PIPELINE_FIELDS:
            self.fields[self.prefix_key + spec[0]] = _pipeline_field(spec, required)

    def pipeline_values(self) -> dict:
        return {spec[0]: self.cleaned_data[self.prefix_key + spec[0]] for spec in PIPELINE_FIELDS
                if self.cleaned_data.get(self.prefix_key + spec[0]) is not None}

    def pipeline_bound_fields(self):
        return [self[self.prefix_key + spec[0]] for spec in PIPELINE_FIELDS]

    def check_ranges(self, merged: dict):
        if merged["period_min"] >= merged["period_max"]:
            self.add_error(self.prefix_key + "period_max", "Maximum period must exceed the minimum.")
        if merged["q_min"] >= merged["q_max"]:
            self.add_error(self.prefix_key + "q_max", "Maximum duty cycle must exceed the minimum.")
        if int(merged["clip_window"]) % 2 == 0:
            self.add_error(self.prefix_key + "clip_window", "Use an odd number of cadences.")


class SettingsForm(PipelineMixin, forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_pipeline_fields(required=True)

    def clean(self):
        data = super().clean()
        if not self.errors:
            self.check_ranges({**DEFAULT_PIPELINE, **self.pipeline_values()})
        return data


class NewRunForm(PipelineMixin, forms.Form):
    prefix_key = "p_"
    SOURCES = [("mast", "MAST archive"), ("synthetic", "Synthetic injection"), ("upload", "CSV upload")]
    KINDS = [("planet", "Transiting planet"), ("eclipsing_binary", "Eclipsing binary"), ("blend", "Background blend"),
             ("noise", "Noise only")]

    source = forms.ChoiceField(choices=SOURCES)
    target_name = forms.CharField(max_length=80, required=False, label="Target name", help_text="Optional label")
    # MAST
    mast_target = forms.CharField(max_length=400, required=False, label="Target",
                                  help_text="TIC / KIC id or name: WASP-12, TIC 261136679. Separate several with commas to queue a batch")
    mission = forms.ChoiceField(choices=[(m, m) for m in ("TESS", "Kepler", "K2")], required=False)
    product = forms.RegexField(r"^\d+:\d+(\.\d+)?$", required=False)
    # synthetic
    kind = forms.ChoiceField(choices=KINDS, required=False, label="Signal type")
    period_days = forms.FloatField(min_value=0.3, max_value=20, required=False, label="Period (days)", help_text="0.3 – 20")
    depth_ppm = forms.FloatField(min_value=1, max_value=249999, required=False, label="Depth (ppm)",
                                 help_text=r"\((R_p/R_\star)^2 \times 10^{6}\); sets companion size for binaries")
    impact_param_syn = forms.FloatField(min_value=0, max_value=1.49, required=False, label="Impact parameter", help_text="0 = central")
    noise_ppm = forms.FloatField(min_value=1, max_value=50000, required=False, label="White noise (ppm)", help_text="Per cadence")
    variability_amp = forms.FloatField(min_value=0, max_value=0.05, required=False, label="Stellar variability", help_text="Fractional amplitude")
    baseline_days = forms.FloatField(min_value=5, max_value=120, required=False, label="Baseline (days)", help_text="5 – 120")
    cadence_min = forms.FloatField(min_value=1, max_value=30, required=False, label="Cadence (min)", help_text="1 – 30")
    seed = forms.IntegerField(min_value=0, required=False, label="Random seed", help_text="Blank = random")
    # upload
    file = forms.FileField(required=False, label="CSV file", help_text="Columns: time, flux and optional flux_err")
    # host star
    t_eff_k = forms.FloatField(min_value=1001, max_value=59999, required=False, label=r"Effective temperature \(T_{\mathrm{eff}}\) (K)")
    r_star_solar = forms.FloatField(min_value=0.011, max_value=199, required=False, label=r"Radius \(R_\star\) (\(R_\odot\))")
    m_star_solar = forms.FloatField(min_value=0.011, max_value=99, required=False, label=r"Mass \(M_\star\) (\(M_\odot\))")

    SYN_DEFAULTS = {"kind": "planet", "period_days": 3.524748, "depth_ppm": 14600, "impact_param_syn": 0.1, "noise_ppm": 1000,
                    "variability_amp": 0.002, "baseline_days": 27.4, "cadence_min": 2}
    SYN_FIELDS = ["kind", "target_name", "period_days", "depth_ppm", "impact_param_syn", "noise_ppm", "variability_amp",
                  "baseline_days", "cadence_min", "seed"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_pipeline_fields(required=False)
        for name, field in self.fields.items():
            if isinstance(field, (forms.FloatField, forms.IntegerField)) and not name.startswith("p_"):
                field.widget.attrs["step"] = "any"
        for key in ("t_eff_k", "r_star_solar", "m_star_solar"):
            self.fields[key].widget.attrs["placeholder"] = "auto"

    def syn_bound_fields(self):
        return [self[name] for name in self.SYN_FIELDS]

    def clean(self):
        data = super().clean()
        if self.errors:
            return data
        source = data["source"]
        config: dict = {"source": source,
                        "stellar_params": {k: data[k] for k in ("t_eff_k", "r_star_solar", "m_star_solar") if data.get(k) is not None},
                        "pipeline": self.pipeline_values()}
        self.check_ranges({**DEFAULT_PIPELINE, **config["pipeline"]})
        if source == "mast":
            if not data.get("mast_target", "").strip():
                self.add_error("mast_target", "Enter a target to fetch from MAST.")
            names = [n.strip() for n in data.get("mast_target", "").replace(";", ",").split(",") if n.strip()]
            if len(names) > 12:
                self.add_error("mast_target", "Queue at most 12 targets at a time.")
            data["batch_targets"] = names
            mast = {"target": names[0] if names else "", "mission": data.get("mission") or "TESS"}
            if data.get("product") and len(names) == 1:
                seq, exptime = data["product"].split(":")
                mast |= {"sequence": int(seq), "exptime": float(exptime)}
            config["mast"] = mast
        elif source == "synthetic":
            v = {k: (data.get(k) if data.get(k) not in (None, "") else d) for k, d in self.SYN_DEFAULTS.items()}
            config["synthetic"] = {"kind": v["kind"], "period_days": v["period_days"], "depth": v["depth_ppm"] / 1e6,
                                   "impact_param": v["impact_param_syn"], "noise_ppm": v["noise_ppm"],
                                   "variability_amp": v["variability_amp"], "baseline_days": v["baseline_days"],
                                   "cadence_min": v["cadence_min"]}
            if data.get("seed") is not None:
                config["synthetic"]["seed"] = data["seed"]
            config["target_name"] = data.get("target_name", "")
        else:
            upload = data.get("file")
            if not upload:
                self.add_error("file", "Choose a CSV file to analyse.")
            else:
                content = upload.read()
                try:
                    parsed = ingest.parse_csv(content)
                except ValueError as exc:
                    self.add_error("file", str(exc))
                else:
                    upload_id = uuid.uuid4().hex
                    (UPLOAD_DIR / f"{upload_id}.csv").write_bytes(content)
                    config["upload_id"] = upload_id
                    config["upload_info"] = {"filename": upload.name, "rows": len(parsed["time"])}
                    config["target_name"] = data.get("target_name") or upload.name.rsplit(".", 1)[0][:80]
        data["config"] = config
        return data


class CalculatorForm(forms.Form):
    r_star_solar = forms.FloatField(min_value=0.011, max_value=199, initial=1.155)
    m_star_solar = forms.FloatField(min_value=0.011, max_value=99, initial=1.119)
    t_eff_k = forms.FloatField(min_value=1001, max_value=59999, initial=6065)
    period_days = forms.FloatField(min_value=0.011, max_value=5000, initial=3.524748)
    depth_ppm = forms.FloatField(min_value=1, max_value=999999, initial=14600)
    impact_param = forms.FloatField(min_value=0, max_value=1.99, initial=0.5)
    bond_albedo = forms.FloatField(min_value=0, max_value=0.99, initial=0.3)


class SignalCheckForm(forms.Form):
    """A signal described by its catalogue parameters. Only the first three are needed."""
    period_days = forms.FloatField(min_value=0.05, max_value=2000, label="Orbital period", help_text="days")
    depth_ppm = forms.FloatField(min_value=1, max_value=999999, label="Transit depth", help_text="parts per million (1% = 10,000)")
    duration_hours = forms.FloatField(min_value=0.05, max_value=100, label="Transit duration", help_text="hours, first to last contact")
    snr = forms.FloatField(min_value=0, max_value=1e6, required=False, label="Signal-to-noise", help_text="of all transits together")
    n_transits = forms.FloatField(min_value=1, max_value=1e5, required=False, label="Transits observed", help_text="how many dips were seen")
    t_eff_k = forms.FloatField(min_value=1001, max_value=59999, required=False, label="Star temperature", help_text="K")
    r_star_solar = forms.FloatField(min_value=0.011, max_value=199, required=False, label="Star radius", help_text="Suns")
    m_star_solar = forms.FloatField(min_value=0.011, max_value=99, required=False, label="Star mass", help_text="Suns")
    odd_even_sigma = forms.FloatField(min_value=0, max_value=1e4, required=False, label="Odd/even depth difference", help_text="in sigma")
    secondary_sigma = forms.FloatField(min_value=-100, max_value=1e4, required=False, label="Secondary eclipse significance", help_text="in sigma, at phase 0.5")
    secondary_depth_ppm = forms.FloatField(min_value=-1e5, max_value=999999, required=False, label="Secondary eclipse depth", help_text="parts per million")
    centroid_sigma = forms.FloatField(min_value=0, max_value=1e4, required=False, label="Centroid shift in transit", help_text="in sigma")

    REQUIRED = ("period_days", "depth_ppm", "duration_hours")

    def groups(self):
        get = lambda *names: [self[n] for n in names]
        return [("The signal", "Needed.", get(*self.REQUIRED)),
                ("How well it was seen", "Optional.", get("snr", "n_transits")),
                ("The star", "Optional. Radius and mass are needed together; they turn the depth into a size.", get("t_eff_k", "r_star_solar", "m_star_solar")),
                ("Vetting measurements", "Optional. These separate planets from eclipsing binaries best.",
                 get("odd_even_sigma", "secondary_sigma", "secondary_depth_ppm", "centroid_sigma"))]

    def clean(self):
        data = super().clean()
        if (data.get("r_star_solar") is None) != (data.get("m_star_solar") is None):
            raise forms.ValidationError("Give the star's radius and mass together, or neither.")
        if data.get("period_days") and data.get("duration_hours") and data["duration_hours"] / 24.0 >= 0.5 * data["period_days"]:
            raise forms.ValidationError("The transit cannot last half the orbit or more.")
        return data


class VettingForm(forms.Form):
    status = forms.ChoiceField(choices=[("pending", "pending"), ("confirmed", "confirmed"), ("false_positive", "false_positive")])
    note = forms.CharField(max_length=2000, required=False, strip=True)
