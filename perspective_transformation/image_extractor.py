import cv2
import numpy as np
import argparse
import os

# Global variables
points = [] # the points we create with mouse clicks
original_image = None
display_image = None


def mouse_callback(event, x, y, flags, param):
    global points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))

            # Draw the clicked point
            cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)

            # If we have 2 or 3 points, just draw lines in click-order temporarily
            if 1 < len(points) < 4:
                cv2.line(display_image, points[-2], points[-1], (0, 255, 0), 2)

            #On the 4th click, sort geometrically and draw the clean rectangle
            elif len(points) == 4:
                # 1. Clear temporary lines by copying original image over
                display_image = original_image.copy()
                
                # 2. Get the correct order
                ordered_pts = order_points(points)
                
                # 3. Redraw the clean points
                for pt in ordered_pts:
                    cv2.circle(display_image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                
                # 4. Draw the closed rectangle contour in correct sequence
                pts_contour = ordered_pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_image, [pts_contour], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.imshow("Input Image", display_image)


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    
    # Sort by X coordinate to get left and right halves
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    # Sort left half by Y to find Top-Left and Bottom-Left
    left_top_sort = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_top_sort[0], left_top_sort[1]
    
    # Sort right half by Y to find Top-Right and Bottom-Right
    right_top_sort = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_top_sort[0], right_top_sort[1]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def warp_perspective(image, pts, width, height):
    rect = order_points(pts)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return warped


def reset_selection():
    global points, display_image
    points = []
    display_image = original_image.copy()
    cv2.imshow("Input Image", display_image)


def main():
    global original_image, display_image

    parser = argparse.ArgumentParser(
        description="Perspective Image Extractor"
    )

    parser.add_argument(
        "input",
        help="Input image path"
    )

    parser.add_argument(
        "output",
        help="Output image path"
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Output width"
    )

    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Output height"
    )

    args = parser.parse_args()

    original_image = cv2.imread(args.input)

    if original_image is None:
        print(f"Error: Could not load image '{args.input}'")
        return

    display_image = original_image.copy()

    cv2.namedWindow("Input Image")
    cv2.setMouseCallback("Input Image", mouse_callback)

    print("Instructions:")
    print("- Click four corner points")
    print("- ESC: Reset selection")
    print("- S: Save warped image (in result window)")
    print("- Q: Quit")

    warped_image = None

    while True:
        cv2.imshow("Input Image", display_image)

        if len(points) == 4 and warped_image is None:
            warped_image = warp_perspective(
                original_image,
                points,
                args.width,
                args.height
            )

            cv2.imshow("Warped Result", warped_image)

        key = cv2.waitKey(20) & 0xFF

        # ESC  reset
        if key == 27:
            warped_image = None
            reset_selection()

            try:
                cv2.destroyWindow("Warped Result")
            except:
                pass

        # Save result
        elif key == ord('s') and warped_image is not None:
            cv2.imwrite(args.output, warped_image)
            print(f"Saved to: {args.output}")

        # Quit
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()