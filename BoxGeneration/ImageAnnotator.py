import cv2, os

class ImageAnnotator:
    def __init__(self):
        self.boxes = []
        self.current_box = []
        self.drawing = False
        self.image = None
        self.exclude_image = False

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box.append((x, y))
            self.boxes.append(tuple(self.current_box))
            self.current_box = []

        if self.drawing:
            clone = self.image.copy()
            cv2.rectangle(clone, self.current_box[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', clone)

    def label_image(self, img_path):
        self.image = cv2.imread(img_path)

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.draw_rectangle)

        save_image = False
        while True:
            clone = self.image.copy()
            for box in self.boxes:
                cv2.rectangle(clone, box[0], box[1], (0, 255, 0), 2)
            # set the name of the window to the image name
            cv2.setWindowTitle('Image', img_path)
            cv2.imshow('Image', clone)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Press 'q' to exit the application
                break
            elif key == ord('r'):  # Press 'r' to reset the bounding boxes
                self.boxes = []
            elif key == ord('s'):  # Press 's' to save the labels
                save_image = True
                break
            elif key == ord('e'):
                print("Excluding image: ", img_path)
                save_image = True
                self.exclude_image = True
                break

        if save_image:
            self.save_label(img_path)
        cv2.destroyAllWindows()
        return save_image

    def save_label(self, img_path):
        output_path = '../outputs/gt_label_viz/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = output_path + img_path.split("/")[-1]
        
        for box in self.boxes:
            cv2.rectangle(self.image, box[0], box[1], (0, 255, 0), 2)
        cv2.imwrite(output_path, self.image)
