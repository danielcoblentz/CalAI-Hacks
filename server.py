from flask import Flask, request, send_file, render_template
from routing import generate_risk_map
import logging


app = Flask(__name__)


# Serve the index.html file
@app.route("/", methods=["GET"])
def index():
   logging.debug("Serving index.html")
   return render_template("index.html")


@app.route("/generate-map", methods=["POST"])
def generate_map():
   try:
       logging.debug("Received request to /generate-map endpoint")
      
       # Parse form data from the request
       origin = request.form.get("origin")
       destination = request.form.get("destination")
       logging.debug(f"Origin: {origin}, Destination: {destination}")
      
       # Validate input
       if not origin or not destination:
           logging.error("Missing origin or destination in request data")
           return ("Missing origin or destination", 400)
      
       # Generate risk map
       output_file = generate_risk_map(origin, destination)
       logging.debug(f"Generated output file: {output_file}")
      
       # Return the file or an error message
       if output_file:
           logging.info("Map generation successful")
           return send_file(output_file)
       else:
           logging.error("Map generation failed")
           return ("Map generation failed", 500)
   except Exception as e:
       logging.exception("An error occurred while processing the request")
       return (f"Internal server error: {str(e)}", 500)


if __name__ == "__main__":
   app.run(debug=True)



