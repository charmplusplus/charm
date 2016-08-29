struct Point {
  double x;
  double y;

  // Returns distance^2, hence the name
  double distance2(const Point& other) {
    return (x - other.x) * (x - other.x)
      + (y - other.y) * (y - other.y);
  }

  void pup(PUP::er &p) {
    p | x;
    p | y;
  }
};
