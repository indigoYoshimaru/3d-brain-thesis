class NiiLabel:
    def __init__(self, color, opacity, smoothness):
        self.actor = None
        self.property = None
        self.smoother = None
        self.color = color
        self.opacity = opacity
        self.smoothness = smoothness


class NiiObject:
    def __init__(self):
        self.file = None
        self.reader = None
        self.extent = ()
        self.labels = []
        self.image_mapper = None
        self.scalar_range = None
