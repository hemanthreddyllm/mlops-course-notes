"""A tiny Flask calculator: the app the course containerises with Docker."""
import os

from flask import Flask, render_template, request

from calculator import OPERATIONS, calculate

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {"a": "", "b": "", "operation": "add"}
    if request.method == "POST":
        form = {k: request.form.get(k, "") for k in form}
        try:
            value = calculate(float(form["a"]), float(form["b"]), form["operation"])
            symbol = OPERATIONS[form["operation"]][0]
            result = f"{form['a']} {symbol} {form['b']} = {value:g}"
        except (ValueError, ZeroDivisionError) as exc:
            error = str(exc)
    return render_template("index.html", result=result, error=error, form=form, operations=OPERATIONS)


@app.route("/health")
def health():
    return {"status": "ok", "hostname": os.uname().nodename}


if __name__ == "__main__":
    # 0.0.0.0 (not 127.0.0.1) so the app is reachable from outside a container
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
