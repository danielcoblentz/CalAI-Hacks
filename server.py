from flask import Flask, request, send_file, render_template
from routing import generate_risk_map
import logging
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.DEBUG)


@app.route("/", methods=["GET"])
def index():
    logging.debug("Serving index.html")
    return render_template("index.html")


@app.route("/generate-map", methods=["POST"])
def generate_map():
    try:
        logging.debug("Received request to /generate-map endpoint")

        # Parse single combined input
        combined = request.form.get("combined")
        if not combined or "," not in combined:
            logging.error("Invalid or missing input format.")
            return ("Invalid input. Please use format: 'Origin, Destination'", 400)

        origin, destination = map(str.strip, combined.split(",", 1))
        logging.debug(f"Origin: {origin}, Destination: {destination}")

        # Generate risk map
        output_file = generate_risk_map(origin, destination)
        logging.debug(f"Generated output file: {output_file}")

        if output_file and os.path.exists(output_file):
            logging.info("Map generation successful")
            return send_file(output_file)
        else:
            logging.error("Map generation failed or file not found")
            return ("Map generation failed", 500)

    except Exception as e:
        logging.exception("An error occurred while processing the request")
        return (f"Internal server error: {str(e)}", 500)


if __name__ == "__main__":
    app.run(debug=True)
